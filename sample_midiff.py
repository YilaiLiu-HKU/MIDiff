"""Sample C-GASF images from a trained MIDiff checkpoint."""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import dist_util, logger
from script_util import (
    NUM_CLASSES,
    midiff_model_and_diffusion_defaults,
    create_midiff_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def _compute_flops_per_step(model, diffusion, batch_size, image_size, image_size_second, class_cond, device):
    """基于单次前向计算 FLOPs per step，使用 fvcore（仅 rank 0 且 CUDA 时尝试）。"""
    if not th.cuda.is_available():
        return None, "CUDA 不可用"
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as e:
        return None, f"未安装 fvcore: {e}（pip install fvcore）"
    shape = (batch_size, 1, image_size, image_size_second)
    x = th.randn(*shape, device=device)
    t = th.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
    t_scaled = diffusion._scale_timesteps(t)
    model_kwargs = {}
    if class_cond:
        model_kwargs["y"] = th.randint(0, NUM_CLASSES, (batch_size,), device=device)

    class _ModelWrapper(th.nn.Module):
        """包装 model 以注入 model_kwargs，供 FlopCountAnalysis 单次前向统计 FLOPs。"""
        def __init__(self, m, kwargs):
            super().__init__()
            self._m = m
            self._kwargs = kwargs
        def forward(self, x, t):
            return self._m(x, t, **self._kwargs)

    try:
        with th.no_grad():
            wrapper = _ModelWrapper(model, model_kwargs)
            wrapper.eval()
            flop_counter = FlopCountAnalysis(wrapper, (x, t_scaled))
            total_flops = flop_counter.total()
        return float(total_flops), None
    except Exception as e:
        return None, f"FlopCountAnalysis 异常: {e}"


def main():
    args = create_argparser().parse_args()

    wall_clock_start = time.perf_counter()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating MIDiff model and diffusion...")
    model, diffusion = create_midiff_model_and_diffusion(
        **args_to_dict(args, midiff_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=True
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print(logger.get_dir())

    num_timesteps = diffusion.num_timesteps
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    image_size_second = args.image_size if args.image_size_second == 0 else args.image_size_second

    if dist.get_rank() == 0:
        logger.log("======== MIDiff sampling setup ========")
        logger.log(f"  world_size: {dist.get_world_size()}")
        logger.log(f"  device: {dist_util.dev()}")
        logger.log(f"  batch_size: {args.batch_size}")
        logger.log(f"  num_samples: {args.num_samples}")
        logger.log(f"  image_shape: (1, {args.image_size}, {image_size_second})")
        logger.log(f"  diffusion_steps (config): {getattr(args, 'diffusion_steps', 'N/A')}")
        logger.log(f"  actual timesteps per sample: {num_timesteps}")
        logger.log(f"  use_ddim: {args.use_ddim}")
        logger.log(f"  save_dir: {args.save_dir}")
        logger.log(f"  model_path: {args.model_path}")
        logger.log("==========================")

    logger.log("sampling MIDiff C-GASF images...")
    sampling_start = time.perf_counter()
    all_images = []
    all_labels = []
    all_traffic_predictions = []
    num_batches_done = 0
    first_batch_elapsed = None
    first_batch_peak_mem_bytes = None

    try:
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            if num_batches_done == 0 and th.cuda.is_available():
                th.cuda.synchronize()
                th.cuda.reset_peak_memory_stats()
            batch_start = time.perf_counter()
            out = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, image_size_second),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = out[0] if isinstance(out, tuple) else out
            traffic_prediction = out[1] if (isinstance(out, tuple) and len(out) > 1) else None
            if th.cuda.is_available():
                th.cuda.synchronize()
            batch_elapsed = time.perf_counter() - batch_start
            if num_batches_done == 0:
                first_batch_elapsed = batch_elapsed
                first_batch_peak_mem_bytes = th.cuda.max_memory_allocated() if th.cuda.is_available() else None
            num_batches_done += 1

            if dist.get_rank() == 0:
                per_step_ms = (batch_elapsed / num_timesteps) * 1000
                total_so_far = (len(all_images) + dist.get_world_size()) * args.batch_size
                logger.log(
                    f"batch {num_batches_done}: {batch_elapsed:.2f}s | "
                    f"step {per_step_ms:.3f}ms | timesteps {num_timesteps} | samples {total_so_far}"
                )
                if num_batches_done == 1 and first_batch_elapsed is not None and first_batch_elapsed > 0:
                    flops_per_step, flops_err = _compute_flops_per_step(
                        model, diffusion, args.batch_size,
                        args.image_size, image_size_second, args.class_cond, dist_util.dev()
                    )
                    if flops_per_step is not None:
                        logger.log(f"  FLOPs per step: {flops_per_step/1e9:.2f} G")
                    else:
                        logger.log(f"  FLOPs per step: unavailable - {flops_err}")

            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)

            all_images.extend([s.cpu().numpy() for s in gathered_samples])

            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            if dist.get_rank() == 0:
                logger.log(f"created {len(all_images) * args.batch_size} samples")

    except KeyboardInterrupt:
        logger.log("\nInterrupted by user. Saving generated samples...")

    sampling_elapsed = time.perf_counter() - sampling_start
    if dist.get_rank() == 0:
        logger.log(f"Sampling wall-clock time: {sampling_elapsed:.2f}s ({sampling_elapsed/60:.2f} min)")
        total_steps = num_batches_done * num_timesteps
        if total_steps > 0:
            per_step_sec = sampling_elapsed / total_steps
            per_step_ms = per_step_sec * 1000
            per_batch_sec = sampling_elapsed / num_batches_done if num_batches_done else 0
            logger.log("======== MIDiff sampling report ========")
            logger.log(f"  总时间步 (total denoising steps): {total_steps}")
            logger.log(f"  time per denoising step: {per_step_ms:.3f} ms ({per_step_sec:.6f} s)")
            logger.log(f"  time per batch: {per_batch_sec:.2f} s")
            logger.log(f"  batches: {num_batches_done}")
            logger.log(f"  total sampling time: {sampling_elapsed:.2f} s ({sampling_elapsed/60:.2f} min)")
            logger.log("==========================")

    arr = None
    if all_images:
        arr = np.concatenate(all_images, axis=0)
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]

        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"{args.model_path.split('/')[-1]}_samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete or interrupted.")

    wall_clock_elapsed = time.perf_counter() - wall_clock_start
    if dist.get_rank() == 0:
        logger.log(f"Total wall-clock time: {wall_clock_elapsed:.2f}s ({wall_clock_elapsed/60:.2f} min)")

    # The rest of the original script (FID evaluation) can be placed here.
    # It will run after a normal completion.
    # On interruption, the script will exit after saving the .npz file.
    # For simplicity, this example just exits cleanly.
    if 'KeyboardInterrupt' in locals():
        sys.exit(0)

    # ====== FID评估 ======
    if dist.get_rank() == 0 and arr is not None:
        try:
            from pytorch_fid import fid_score
        except ImportError:
            print("请先 pip install pytorch-fid")
            return
        sampled_dir = os.path.join(args.save_dir, "sampled_images")
        os.makedirs(sampled_dir, exist_ok=True)

        # 保存最大值预测到CSV文件
        csv_path = os.path.join(args.save_dir, f"{args.model_path.split('/')[-1]}_max_predictions.csv")
        """with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_index', 'max_predictions'])  # 写入表头
            for i, max_pred in enumerate(max_traffic_predictions_arr):
                row = [i] + max_pred.tolist()  # 将数组转换为列表并与索引组合
                writer.writerow(row)
        logger.log(f"保存最大值预测到: {csv_path}")"""

        # 保存裁剪后的图像用于FID计算
        for i, img_float in enumerate(arr): # arr 包含的是 [-1, 1] 范围的 float numpy 数组

            # --- 对齐 train_util.py 的处理逻辑 ---
            # 1. 将 [-1, 1] 转换为 [0, 255] uint8
            img_uint8 = ((img_float + 1) * 127.5).clip(0, 255).astype(np.uint8)

            # 2. Squeeze 掉单通道维度
            img_squeezed = np.squeeze(img_uint8)

            # 3. 使用 PIL 以 'L' 模式保存
            # 注意：arr 已经是裁剪过的了，这里直接保存
            save_path = os.path.join(sampled_dir, f"{i:05d}.png")
            Image.fromarray(img_squeezed, mode='L').save(save_path)
        real_dir = str(REPO_ROOT / "dataset_tiff_visual")
        # 处理真实图片：中心裁剪到288*140

        # 计算FID（使用裁剪后的图片）
        """fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, sampled_dir],
            batch_size=args.num_samples,
            device=th.device("cuda" if th.cuda.is_available() else "cpu"),
            dims=2048,
        )
        print(f"FID: {fid_value}")
        logger.log(f"FID: {fid_value}")"""


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=3000,
        batch_size=64,
        use_ddim=False,
        model_path=str(REPO_ROOT / "ckpt" / "midiff" / "ema_0.9999_048000.pt"),
        save_dir=str(REPO_ROOT / "ckpt" / "midiff"),
        image_size_second=160,
        attention_type='triple',
    )
    model_defaults = midiff_model_and_diffusion_defaults()
    model_defaults.update(defaults)
    defaults = model_defaults
    parser = argparse.ArgumentParser(description="Sample C-GASF images with MIDiff.")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

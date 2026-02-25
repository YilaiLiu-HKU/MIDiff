"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

import dist_util, logger
from script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    wall_clock_start = time.perf_counter()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False
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
        logger.log("======== 采样部署 ========")
        logger.log(f"  world_size: {dist.get_world_size()}")
        logger.log(f"  device: {dist_util.dev()}")
        logger.log(f"  batch_size: {args.batch_size}")
        logger.log(f"  num_samples: {args.num_samples}")
        logger.log(f"  image_shape: (1, {args.image_size}, {image_size_second})")
        logger.log(f"  diffusion_steps (config): {getattr(args, 'diffusion_steps', 'N/A')}")
        logger.log(f"  实际每样本时间步: {num_timesteps}")
        logger.log(f"  use_ddim: {args.use_ddim}")
        logger.log(f"  save_dir: {args.save_dir}")
        logger.log(f"  model_path: {args.model_path}")
        logger.log("==========================")

    logger.log("sampling...")
    sampling_start = time.perf_counter()
    all_images = []
    all_labels = []
    all_traffic_predictions = []
    num_batches_done = 0

    try:
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            batch_start = time.perf_counter()
            sample, traffic_prediction = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, image_size_second),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            batch_elapsed = time.perf_counter() - batch_start
            num_batches_done += 1

            if dist.get_rank() == 0:
                per_step_ms = (batch_elapsed / num_timesteps) * 1000
                total_so_far = (len(all_images) + dist.get_world_size()) * args.batch_size
                logger.log(
                    f"batch {num_batches_done}: {batch_elapsed:.2f}s | "
                    f"单步 {per_step_ms:.3f}ms | 时间步 {num_timesteps} | 累计样本 {total_so_far}"
                )

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
            logger.log("======== 采样报告 ========")
            logger.log(f"  总时间步 (total denoising steps): {total_steps}")
            logger.log(f"  单步计算时间: {per_step_ms:.3f} ms ({per_step_sec:.6f} s)")
            logger.log(f"  每 batch 时间: {per_batch_sec:.2f} s")
            logger.log(f"  batch 数: {num_batches_done}")
            logger.log(f"  总采样时间: {sampling_elapsed:.2f} s ({sampling_elapsed/60:.2f} min)")
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
        real_dir = "/home/yilai/poster/NetDiffus/dataset_tiff_visual"
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
        num_samples=5,
        batch_size=64,
        use_ddim=False,
        model_path="",
        save_dir='128/iterate/df/synth_models',
        image_size_second=0,
        attention_type='origin',
        backbone_type='resnet',
        backbone='unet',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

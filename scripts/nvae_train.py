"""
Train NVAE for ablation studies.
Supports non-square images (256x160).
"""

import argparse
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from PIL import Image
import blobfile as bf

import dist_util, logger
from image_datasets import load_data
from script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from nvae_modules import NVAE


def parse_resume_step_from_filename(filename):
    """Parse filenames of the form path/to/nvae_model_NNNNNN.pt"""
    split = filename.split("nvae_model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def sample_and_save_nvae(ddp_model, args, step, num_samples_to_save=5):
    """采样并保存图像"""
    ddp_model.eval()
    
    with th.no_grad():
        sample = ddp_model.module.sample(num_samples_to_save, dist_util.dev())
    
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1).contiguous()
    
    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample)
    
    if dist.get_rank() == 0:
        arr = np.concatenate([s.cpu().numpy() for s in gathered_samples], axis=0)
        arr = arr[:num_samples_to_save]
        
        sampled_dir = os.path.join(logger.get_dir(), "samples_nvae_during_train")
        os.makedirs(sampled_dir, exist_ok=True)
        
        for i, img_np_hwc in enumerate(arr):
            img_np_hw = np.squeeze(img_np_hwc, axis=-1)
            out_path_png = os.path.join(sampled_dir, f"sample_{step:06d}_{i:02d}.png")
            Image.fromarray(img_np_hw, mode='L').save(out_path_png)
        
        logger.log(f"saved {num_samples_to_save} samples to {sampled_dir}")
    
    ddp_model.train()
    dist.barrier()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating NVAE model...")
    model = NVAE(
        in_channels=1,
        base_channels=args.base_channels,
        num_scales=args.num_scales,
        num_cells_per_scale=args.num_cells_per_scale,
        latent_dim=args.latent_dim,
        img_size_h=args.image_size_h,
        img_size_w=args.image_size_w
    ).to(dist_util.dev())
    
    # Checkpoint 加载
    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(f"loading NVAE model from checkpoint: {args.resume_checkpoint}...")
        model.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint, map_location=dist_util.dev())
        )
    
    ddp_model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size_h,
        class_cond=False,
        random_crop=False
    )

    logger.log("creating optimizer...")
    opt = AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 加载 Optimizer 状态
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt_nvae_{resume_step:06d}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}...")
            opt.load_state_dict(
                dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            )

    logger.log("training NVAE...")
    
    step = 0
    total_steps = args.lr_anneal_steps or 1
    while step + resume_step < total_steps:
        batch, _, _, _ = next(data)
        real_imgs = batch.to(dist_util.dev())
        
        if real_imgs.shape[2] != args.image_size_h or real_imgs.shape[3] != args.image_size_w:
            real_imgs = F.interpolate(
                real_imgs,
                size=(args.image_size_h, args.image_size_w),
                mode='bilinear',
                align_corners=False
            )
        
        recons, kl_losses = ddp_model(real_imgs)
        
        # 重建损失
        recons_loss = F.mse_loss(recons, real_imgs)
        
        # 总 KL 损失（多尺度加权）
        kl_loss_total = sum(kl_losses)
        
        # 总损失
        loss = recons_loss + args.kl_weight * kl_loss_total
        
        opt.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        th.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.grad_clip)
        
        opt.step()
        
        current_step = step + resume_step
        
        if current_step % args.log_interval == 0:
            logger.logkv("step", current_step)
            logger.logkv("loss", loss.item())
            logger.logkv("recons_loss", recons_loss.item())
            logger.logkv("kl_loss", kl_loss_total.item())
            for i, kl in enumerate(kl_losses):
                logger.logkv(f"kl_loss_scale_{i}", kl.item())
            logger.dumpkvs()
        
        if current_step > 0 and current_step % args.save_interval == 0:
            if dist.get_rank() == 0:
                logger.log(f"saving model at step {current_step}...")
                model_path = os.path.join(logger.get_dir(), f"nvae_model_{current_step:06d}.pt")
                opt_path = os.path.join(logger.get_dir(), f"opt_nvae_{current_step:06d}.pt")
                th.save(ddp_model.module.state_dict(), model_path)
                th.save(opt.state_dict(), opt_path)
            
            logger.log("saving training samples...")
            sample_and_save_nvae(ddp_model, args, current_step)
        
        step += 1


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.pop('image_size', None)
    
    defaults.update(dict(
        image_size_h=256,
        image_size_w=160,
        lr=1e-3,
        weight_decay=3e-4,
        batch_size=16,  # NVAE 通常使用较小的 batch size
        lr_anneal_steps=300000,
        save_dir="/data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr_nvae_ablation_256x160",
        latent_dim=256,
        base_channels=64,
        num_scales=3,
        num_cells_per_scale=2,
        kl_weight=1e-4,  # NVAE 的 KL 权重通常较小
        grad_clip=1.0,
        data_dir="/home/yilai/projects/poster/NetDiffus/tiff_log_thr",
        resume_checkpoint="",
        log_interval=100,
        save_interval=5000,
    ))
    
    # 移除不需要的参数
    for key in ['num_channels', 'num_res_blocks', 'attention_type',
                'attention_resolutions', 'channel_mult', 'use_scale_shift_norm',
                'dropout', 'use_checkpoint', 'num_heads', 'num_head_channels',
                'use_fp16', 'resblock_updown', 'use_new_attention_order']:
        defaults.pop(key, None)
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

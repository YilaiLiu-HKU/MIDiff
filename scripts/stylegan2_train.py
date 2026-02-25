"""
Train StyleGAN2 for ablation studies.
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
from torch.optim import Adam
from PIL import Image
import blobfile as bf

import dist_util, logger
from image_datasets import load_data
from script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from stylegan2_modules import StyleGAN2Generator, StyleGAN2Discriminator


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/stylegan2_G_NNNNNN.pt
    """
    split = filename.split("stylegan2_G_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def compute_gradient_penalty(D, real_samples, fake_samples):
    """计算梯度惩罚 (R1 regularization)"""
    real_samples.requires_grad_(True)
    real_validity = D(real_samples)
    
    grad_outputs = th.ones_like(real_validity)
    gradients = th.autograd.grad(
        outputs=real_validity,
        inputs=real_samples,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) ** 2)).mean()
    return gradient_penalty


def sample_and_save_stylegan2(ddp_G, args, step, num_samples_to_save=5):
    """采样并保存图像"""
    ddp_G.eval()
    
    z = th.randn(num_samples_to_save, args.latent_dim, device=dist_util.dev())
    
    with th.no_grad():
        sample = ddp_G.module(z)
    
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1).contiguous()
    
    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample)
    
    if dist.get_rank() == 0:
        arr = np.concatenate([s.cpu().numpy() for s in gathered_samples], axis=0)
        arr = arr[:num_samples_to_save]
        
        sampled_dir = os.path.join(logger.get_dir(), "samples_stylegan2_during_train")
        os.makedirs(sampled_dir, exist_ok=True)
        
        for i, img_np_hwc in enumerate(arr):
            img_np_hw = np.squeeze(img_np_hwc, axis=-1)
            out_path_png = os.path.join(sampled_dir, f"sample_{step:06d}_{i:02d}.png")
            Image.fromarray(img_np_hw, mode='L').save(out_path_png)
        
        logger.log(f"saved {num_samples_to_save} samples to {sampled_dir}")
    
    ddp_G.train()
    dist.barrier()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating StyleGAN2 models...")
    
    G = StyleGAN2Generator(
        z_dim=args.latent_dim,
        w_dim=args.w_dim,
        img_channels=1,
        img_size_h=args.image_size_h,
        img_size_w=args.image_size_w
    ).to(dist_util.dev())
    
    D = StyleGAN2Discriminator(
        img_channels=1,
        img_size_h=args.image_size_h,
        img_size_w=args.image_size_w
    ).to(dist_util.dev())
    
    # --- Checkpoint 加载 ---
    resume_step = 0
    if args.resume_checkpoint_g:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint_g)
        if dist.get_rank() == 0:
            logger.log(f"loading Generator from checkpoint: {args.resume_checkpoint_g}...")
        G.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint_g, map_location=dist_util.dev())
        )
    
    if args.resume_checkpoint_d:
        if dist.get_rank() == 0:
            logger.log(f"loading Discriminator from checkpoint: {args.resume_checkpoint_d}...")
        D.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint_d, map_location=dist_util.dev())
        )
    # -----------------------
    
    ddp_G = DDP(
        G,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    ddp_D = DDP(
        D,
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

    logger.log("creating optimizers...")
    opt_G = Adam(ddp_G.parameters(), lr=args.lr_g, betas=(0.0, 0.99))
    opt_D = Adam(ddp_D.parameters(), lr=args.lr_d, betas=(0.0, 0.99))
    
    # --- 加载 Optimizer 状态 ---
    if args.resume_checkpoint_g:
        opt_g_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint_g), f"opt_G_{resume_step:06d}.pt"
        )
        if bf.exists(opt_g_checkpoint):
            logger.log(f"loading Generator optimizer state from checkpoint: {opt_g_checkpoint}...")
            opt_G.load_state_dict(
                dist_util.load_state_dict(opt_g_checkpoint, map_location=dist_util.dev())
            )
        else:
            logger.log(f"Generator optimizer checkpoint {opt_g_checkpoint} not found, starting fresh.")
    
    if args.resume_checkpoint_d:
        opt_d_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint_d), f"opt_D_{resume_step:06d}.pt"
        )
        if bf.exists(opt_d_checkpoint):
            logger.log(f"loading Discriminator optimizer state from checkpoint: {opt_d_checkpoint}...")
            opt_D.load_state_dict(
                dist_util.load_state_dict(opt_d_checkpoint, map_location=dist_util.dev())
            )
        else:
            logger.log(f"Discriminator optimizer checkpoint {opt_d_checkpoint} not found, starting fresh.")
    # -------------------------

    logger.log("training StyleGAN2...")
    
    step = 0
    while step + resume_step < args.lr_anneal_steps or not args.lr_anneal_steps:
        # ---------------------
        #  训练判别器 (D)
        # ---------------------
        for _ in range(args.d_steps):
            opt_D.zero_grad()
            
            batch, _, _, _ = next(data)
            real_imgs = batch.to(dist_util.dev())
            
            if real_imgs.shape[2] != args.image_size_h or real_imgs.shape[3] != args.image_size_w:
                real_imgs = F.interpolate(
                    real_imgs,
                    size=(args.image_size_h, args.image_size_w),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Wasserstein loss
            real_validity = ddp_D(real_imgs).mean()
            
            z = th.randn(real_imgs.size(0), args.latent_dim, device=dist_util.dev())
            fake_imgs = ddp_G(z)
            fake_validity = ddp_D(fake_imgs.detach()).mean()
            
            # R1 梯度惩罚
            gp = compute_gradient_penalty(ddp_D, real_imgs, fake_imgs) if step % args.gp_interval == 0 else 0
            
            d_loss = fake_validity - real_validity + args.gp_lambda * gp
            d_loss.backward()
            opt_D.step()

        # ---------------------
        #  训练生成器 (G)
        # ---------------------
        opt_G.zero_grad()
        
        z = th.randn(args.batch_size, args.latent_dim, device=dist_util.dev())
        gen_imgs = ddp_G(z)
        g_loss = -ddp_D(gen_imgs).mean()
        
        g_loss.backward()
        opt_G.step()

        current_step = step + resume_step
        
        if current_step % args.log_interval == 0:
            logger.logkv("step", current_step)
            logger.logkv("d_loss", d_loss.item())
            logger.logkv("g_loss", g_loss.item())
            if gp != 0:
                logger.logkv("gp", gp.item())
            logger.dumpkvs()
            
        if current_step % args.save_interval == 0 and current_step > 0:
            if dist.get_rank() == 0:
                logger.log(f"saving models at step {current_step}...")
                g_path = os.path.join(logger.get_dir(), f"stylegan2_G_{current_step:06d}.pt")
                d_path = os.path.join(logger.get_dir(), f"stylegan2_D_{current_step:06d}.pt")
                opt_g_path = os.path.join(logger.get_dir(), f"opt_G_{current_step:06d}.pt")
                opt_d_path = os.path.join(logger.get_dir(), f"opt_D_{current_step:06d}.pt")
                
                th.save(ddp_G.module.state_dict(), g_path)
                th.save(ddp_D.module.state_dict(), d_path)
                th.save(opt_G.state_dict(), opt_g_path)
                th.save(opt_D.state_dict(), opt_d_path)
            
            logger.log("saving training samples...")
            sample_and_save_stylegan2(ddp_G, args, current_step)
            
            dist.barrier()

        step += 1


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.pop('image_size', None)
    
    defaults.update(dict(
        image_size_h=256,
        image_size_w=160,
        lr_g=2e-3,
        lr_d=2e-3,
        batch_size=16,  # StyleGAN2 通常使用较小的 batch size
        lr_anneal_steps=300000,
        save_dir="/data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr_stylegan2_ablation_256x160",
        latent_dim=256,
        w_dim=512,
        data_dir="/home/yilai/projects/poster/NetDiffus/tiff_log_thr",
        d_steps=1,  # 每个 G step 训练 D 的次数
        gp_lambda=10.0,  # R1 正则化系数
        gp_interval=4,  # 梯度惩罚计算间隔
        log_interval=100,
        save_interval=10625,
        resume_checkpoint_g="",  # Generator checkpoint path
        resume_checkpoint_d="",  # Discriminator checkpoint path
    ))
    
    # 移除不需要的参数
    for key in ['lr', 'num_channels', 'num_res_blocks', 'attention_type', 
                'attention_resolutions', 'channel_mult', 'use_scale_shift_norm',
                'dropout', 'use_checkpoint', 'num_heads', 'num_head_channels',
                'use_fp16', 'resblock_updown', 'use_new_attention_order']:
        defaults.pop(key, None)
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

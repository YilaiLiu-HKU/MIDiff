"""
Train a Generative Adversarial Network (GAN) for ablation studies.
Uses DecoderUNetModel as Generator and moe_discriminator.Discriminator as Discriminator.

** Updated (v2) to natively support non-square images (e.g., 256x160) 
** WITHOUT using F.interpolate.
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

# 复用您现有的工具
import dist_util, logger
from image_datasets import load_data
from script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)

# 导入 GAN 组件
from unet_new import DecoderUNetModel # 作为 Generator
from moe_discriminator import Discriminator # 作为 Discriminator
from nn import timestep_embedding


def parse_resume_step_from_filename(filename):
    """Parse filenames of the form path/to/gan_G_NNNNNN.pt"""
    split = filename.split("gan_G_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def sample_and_save_gan(ddp_G, args, step, num_samples_to_save=5):
    """采样并保存图像"""
    ddp_G.eval()
    
    z = th.randn(num_samples_to_save, args.latent_dim, device=dist_util.dev())
    
    with th.no_grad():
        sample = ddp_G(z)
    
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1).contiguous()
    
    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample)
    
    if dist.get_rank() == 0:
        arr = np.concatenate([s.cpu().numpy() for s in gathered_samples], axis=0)
        arr = arr[:num_samples_to_save]
        
        sampled_dir = os.path.join(logger.get_dir(), "samples_gan_during_train")
        os.makedirs(sampled_dir, exist_ok=True)
        
        for i, img_np_hwc in enumerate(arr):
            img_np_hw = np.squeeze(img_np_hwc, axis=-1)
            out_path_png = os.path.join(sampled_dir, f"sample_{step:06d}_{i:02d}.png")
            Image.fromarray(img_np_hw, mode='L').save(out_path_png)
        
        logger.log(f"saved {num_samples_to_save} samples to {sampled_dir}")
    
    ddp_G.train()
    dist.barrier()


class Generator(nn.Module):
    """
    GAN Generator wrapper for DecoderUNetModel.
    Natively handles non-square bottlenecks.
    """
    def __init__(self, args, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size_h = args.image_size_h
        self.image_size_w = args.image_size_w
        args.image_size=256
        model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
        
        # 1. 初始化解码器 (Decoder)
        #
        self.decoder = DecoderUNetModel(
            image_size=self.image_size_h,
            out_channels=1, 
            model_channels=model_args['num_channels'],
            num_res_blocks=model_args['num_res_blocks'],
            attention_resolutions=tuple(int(res) for res in model_args['attention_resolutions'].split(",")),
            latent_dim=latent_dim,
            dropout=model_args['dropout'],
            channel_mult=tuple(int(ch) for ch in model_args['channel_mult'].split(",")),
            dims=2,
            use_checkpoint=model_args['use_checkpoint'],
            num_heads=model_args['num_heads'],
            num_head_channels=model_args['num_head_channels'],
            use_scale_shift_norm=model_args['use_scale_shift_norm'],
            resblock_updown=model_args['resblock_updown'],
            use_fp16=model_args['use_fp16'],
        )
        
        # --- 关键修改：重写(Override)解码器的瓶颈层 ---
        self.decoder_channel_mult = tuple(int(ch) for ch in model_args['channel_mult'].split(","))
        self.decoder_model_channels = model_args['num_channels']
        
        num_downsamples = len(self.decoder_channel_mult) - 1
        self.bottleneck_res_h = self.image_size_h // (2 ** num_downsamples)
        self.bottleneck_res_w = self.image_size_w // (2 ** num_downsamples)
        
        bottleneck_ch = self.decoder_model_channels * self.decoder_channel_mult[-1]
        
        # 覆盖 latent_proj 层
        self.decoder.latent_proj = nn.Linear(
            latent_dim, 
            bottleneck_ch * self.bottleneck_res_h * self.bottleneck_res_w
        )
        self.final_act = nn.Tanh()

    def forward(self, z):
        # GAN的G通常不需要时间步，我们传递一个0张量
        t_zeros = th.zeros(z.shape[0], device=z.device, dtype=th.long)
        
        # 解码 (重写解码器的 forward 逻辑以使用非正方形瓶颈)
        #
        emb = self.decoder.time_embed(
            timestep_embedding(t_zeros, self.decoder_model_channels)
        )
        
        h_dec = self.decoder.latent_proj(z)
        
        h_dec = h_dec.view(
            h_dec.shape[0], 
            self.decoder_model_channels * self.decoder_channel_mult[-1], 
            self.bottleneck_res_h, 
            self.bottleneck_res_w
        )
        
        h_dec = self.decoder.middle_block(h_dec, emb)
        
        for module in self.decoder.output_blocks:
            h_dec = module(h_dec, emb)
            
        h_dec = h_dec.type(z.dtype)
        gen_img = self.decoder.out(h_dec)
        
        return self.final_act(gen_img)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating GAN models (G and D) (non-square)...")
    
    G = Generator(args, args.latent_dim).to(dist_util.dev())
    D = Discriminator(channels=1).to(dist_util.dev())
    
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
    #
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size_h, # 256
        class_cond=False, 
        random_crop=False # ** 关键：确保不裁剪成正方形 **
    )

    logger.log("creating optimizers...")
    opt_G = Adam(ddp_G.parameters(), lr=args.lr_g, betas=(args.b1, args.b2))
    opt_D = Adam(ddp_D.parameters(), lr=args.lr_d, betas=(args.b1, args.b2))
    
    # --- 加载 Optimizer 状态 ---
    if args.resume_checkpoint_g:
        opt_g_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint_g), f"opt_gan_G_{resume_step:06d}.pt"
        )
        if bf.exists(opt_g_checkpoint):
            logger.log(f"loading Generator optimizer state from checkpoint: {opt_g_checkpoint}...")
            opt_G.load_state_dict(
                dist_util.load_state_dict(opt_g_checkpoint, map_location=dist_util.dev())
            )
    
    if args.resume_checkpoint_d:
        opt_d_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint_d), f"opt_gan_D_{resume_step:06d}.pt"
        )
        if bf.exists(opt_d_checkpoint):
            logger.log(f"loading Discriminator optimizer state from checkpoint: {opt_d_checkpoint}...")
            opt_D.load_state_dict(
                dist_util.load_state_dict(opt_d_checkpoint, map_location=dist_util.dev())
            )
    # -------------------------
    
    adversarial_loss = nn.BCELoss().to(dist_util.dev()) 

    logger.log("training GAN...")
    
    step = 0
    while step + resume_step < args.lr_anneal_steps or not args.lr_anneal_steps:
        current_step = step + resume_step
        if current_step % 100 == 0:
            logger.log(f"step: {current_step}")
        # ---------------------
        #  训练判别器 (D)
        # ---------------------
        
        opt_D.zero_grad()
        
        batch, _, _, _ = next(data)
        real_imgs = batch.to(dist_util.dev()) # [B, 1, 256, 160]
        
        # ** 移除 F.interpolate **
        
        # 检查尺寸
        if real_imgs.shape[2] != args.image_size_h or real_imgs.shape[3] != args.image_size_w:
            real_imgs = F.interpolate(
                real_imgs, 
                size=(args.image_size_h, args.image_size_w), 
                mode='bilinear', 
                align_corners=False
            )

        valid = th.full((real_imgs.size(0), 1), 1.0, device=dist_util.dev(), dtype=th.float32)
        fake = th.full((real_imgs.size(0), 1), 0.0, device=dist_util.dev(), dtype=th.float32)

        real_loss = adversarial_loss(ddp_D(real_imgs), valid)
        
        z = th.randn(real_imgs.size(0), args.latent_dim, device=dist_util.dev())
        gen_imgs = ddp_G(z)
        
        fake_loss = adversarial_loss(ddp_D(gen_imgs.detach()), fake)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()

        # ---------------------
        #  训练生成器 (G)
        # ---------------------
        
        opt_G.zero_grad()
        
        g_loss = adversarial_loss(ddp_D(gen_imgs), valid)
        
        g_loss.backward()
        opt_G.step()

        if current_step % 1000 == 0:
            logger.logkv("step", current_step)
            logger.logkv("d_loss", d_loss.item())
            logger.logkv("g_loss", g_loss.item())
            logger.dumpkvs()
            
        if current_step % 10625 == 0 and current_step > 0:
            if dist.get_rank() == 0:
                logger.log(f"saving models at step {current_step}...")
                g_path = os.path.join(logger.get_dir(), f"gan_G_{current_step:06d}.pt")
                d_path = os.path.join(logger.get_dir(), f"gan_D_{current_step:06d}.pt")
                opt_g_path = os.path.join(logger.get_dir(), f"opt_gan_G_{current_step:06d}.pt")
                opt_d_path = os.path.join(logger.get_dir(), f"opt_gan_D_{current_step:06d}.pt")
                
                th.save(ddp_G.module.state_dict(), g_path)
                th.save(ddp_D.module.state_dict(), d_path)
                th.save(opt_G.state_dict(), opt_g_path)
                th.save(opt_D.state_dict(), opt_d_path)
            
            logger.log("saving training samples...")
            sample_and_save_gan(ddp_G, args, current_step)
            
            dist.barrier()

        step += 1

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.pop('image_size', None) # 移除旧的 image_size
    
    defaults.update(dict(
        image_size_h=256,
        image_size_w=160,
        lr_g=2e-4, 
        lr_d=2e-4, 
        b1=0.5,
        b2=0.999,
        batch_size=32,
        lr_anneal_steps=300000, 
        save_dir="/data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr_gan_ablation_256x160",
        latent_dim=256, 
        data_dir="/home/yilai/projects/poster/NetDiffus/tiff_log_thr",
        # 保持与您的扩散模型指令一致
        num_channels=128, 
        num_res_blocks=3, 
        attention_type='triple', 
        attention_resolutions="32,16,8",
        channel_mult="1,1,2,3,4", # 必须提供这个来计算瓶颈
        use_scale_shift_norm=True,
        resume_checkpoint_g="",  # Generator checkpoint path
        resume_checkpoint_d="",  # Discriminator checkpoint path
    ))
    defaults.pop('lr', None)
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
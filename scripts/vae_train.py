"""
Train a Variational Autoencoder (VAE) for ablation studies.
Uses EncoderUNetModel and DecoderUNetModel to maintain backbone consistency.

** Updated (v4) to:
** 1. Natively support non-square images (256x160).
** 2. Add nn.Tanh() for [-1, 1] output.
** 3. Add sample saving during training intervals (5 un-cropped images).
** 4. Add support for --resume_checkpoint.
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
import blobfile as bf #

# 复用您现有的工具
import dist_util, logger
from image_datasets import load_data
from losses import normal_kl
from script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
# 导入 VAE 组件
from unet import EncoderUNetModel
from unet_new import DecoderUNetModel
from nn import timestep_embedding

# --- Helper Function (from train_util.py) ---
#
def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/vae_model_NNNNNN.pt
    """
    split = filename.split("vae_model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
# --- End Helper Function ---

def reparameterize(mu, logvar):
    """ Reparameterization trick """
    std = th.exp(0.5 * logvar)
    eps = th.randn_like(std)
    return mu + eps * std

class VAE(nn.Module):
    """
    VAE wrapper - 必须与 vae_train.py 中的 VAE 类完全一致
    """
    def __init__(self, args, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size_h = args.image_size_h
        self.image_size_w = args.image_size_w
        args.image_size=256
        encoder_args = args_to_dict(args, model_and_diffusion_defaults().keys())
        
        # 1. 初始化编码器 (Encoder)
        self.encoder = EncoderUNetModel(
            image_size=self.image_size_h, 
            in_channels=1, 
            model_channels=encoder_args['num_channels'],
            out_channels=latent_dim * 2, 
            num_res_blocks=encoder_args['num_res_blocks'],
            attention_resolutions=tuple(int(res) for res in encoder_args['attention_resolutions'].split(",")),
            dropout=encoder_args['dropout'],
            channel_mult=tuple(int(ch) for ch in encoder_args['channel_mult'].split(",")),
            use_checkpoint=encoder_args['use_checkpoint'],
            use_fp16=encoder_args['use_fp16'],
            num_heads=encoder_args['num_heads'],
            num_head_channels=encoder_args['num_head_channels'],
            use_scale_shift_norm=encoder_args['use_scale_shift_norm'],
            resblock_updown=encoder_args['resblock_updown'],
            use_new_attention_order=encoder_args['use_new_attention_order'],
            pool="adaptive"
        )

        # 2. 初始化解码器 (Decoder)
        self.decoder = DecoderUNetModel(
            image_size=self.image_size_h,
            out_channels=1, 
            model_channels=encoder_args['num_channels'],
            num_res_blocks=encoder_args['num_res_blocks'],
            attention_resolutions=tuple(int(res) for res in encoder_args['attention_resolutions'].split(",")),
            latent_dim=latent_dim,
            dropout=encoder_args['dropout'],
            channel_mult=tuple(int(ch) for ch in encoder_args['channel_mult'].split(",")),
            dims=2,
            use_checkpoint=encoder_args['use_checkpoint'],
            num_heads=encoder_args['num_heads'],
            num_head_channels=encoder_args['num_head_channels'],
            use_scale_shift_norm=encoder_args['use_scale_shift_norm'],
            resblock_updown=encoder_args['resblock_updown'],
            use_fp16=encoder_args['use_fp16'],
        )
        
        # 3. 重写瓶颈层
        self.decoder_channel_mult = tuple(int(ch) for ch in encoder_args['channel_mult'].split(","))
        self.decoder_model_channels = encoder_args['num_channels']
        
        num_downsamples = len(self.decoder_channel_mult) - 1
        self.bottleneck_res_h = self.image_size_h // (2 ** num_downsamples)
        self.bottleneck_res_w = self.image_size_w // (2 ** num_downsamples)
        bottleneck_ch = self.decoder_model_channels * self.decoder_channel_mult[-1]
        
        self.decoder.latent_proj = nn.Linear(
            latent_dim, 
            bottleneck_ch * self.bottleneck_res_h * self.bottleneck_res_w
        )
        
        # 4. Tanh 激活函数 (修正)
        self.final_act = nn.Tanh()

    def forward(self, x, sample_z=None):
        if sample_z is not None:
            return self.decode(sample_z)

        t_zeros = th.zeros(x.shape[0], device=x.device, dtype=th.long)
        h = self.encoder(x, timesteps=t_zeros)
        mu, logvar = th.split(h, self.latent_dim, dim=1)
        z = reparameterize(mu, logvar)
        recons = self.decode(z, t_zeros)
        return recons, mu, logvar

    def decode(self, z, t_zeros=None):
        """ 解码器部分 """
        if t_zeros is None:
            t_zeros = th.zeros(z.shape[0], device=z.device, dtype=th.long)
            
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
        recons = self.decoder.out(h_dec)
        return self.final_act(recons)

def sample_and_save_vae(ddp_vae_model, args, step, num_samples_to_save=5):
    """ 在训练间隔中采样并保存图像 (无裁剪) """
    ddp_vae_model.eval() 
    z = th.randn(num_samples_to_save, args.latent_dim, device=dist_util.dev())
    
    with th.no_grad():
        sample = ddp_vae_model.module.decode(z) 

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1).contiguous() 

    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample)
    
    if dist.get_rank() == 0:
        arr = np.concatenate([s.cpu().numpy() for s in gathered_samples], axis=0)
        arr = arr[:num_samples_to_save] 

        sampled_dir = os.path.join(logger.get_dir(), "samples_vae_during_train")
        os.makedirs(sampled_dir, exist_ok=True)
        
        for i, img_np_hwc in enumerate(arr):
            img_np_hw = np.squeeze(img_np_hwc, axis=-1)
            out_path_png = os.path.join(sampled_dir, f"sample_{step:06d}_{i:02d}.png")
            Image.fromarray(img_np_hw, mode='L').save(out_path_png)
        
        logger.log(f"saved {num_samples_to_save} un-cropped samples to {sampled_dir}")

    ddp_vae_model.train() 
    dist.barrier()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating VAE model (non-square)...")
    vae = VAE(args, args.latent_dim).to(dist_util.dev())
    
    # --- Checkpoint 加载 ---
    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(f"loading VAE model from checkpoint: {args.resume_checkpoint}...")
        vae.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint, map_location=dist_util.dev())
        )
    # -----------------------

    ddp_vae = DDP(
        vae,
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
    opt = AdamW(ddp_vae.parameters(), lr=args.lr, weight_decay=0)

    # --- 加载 Optimizer 状态 ---
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt_vae_{resume_step:06d}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading VAE optimizer state from checkpoint: {opt_checkpoint}...")
            opt.load_state_dict(
                dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            )
        else:
            logger.log(f"Optimizer checkpoint {opt_checkpoint} not found, starting fresh.")
    # -------------------------

    logger.log("training VAE...")
    
    step = 0
    total_steps = args.lr_anneal_steps or 1 # 确保至少运行1步
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

        recons, mu, logvar = ddp_vae(real_imgs)
        
        recons_loss = F.mse_loss(recons, real_imgs)
        kl_loss = normal_kl(mu, logvar, 0.0, 0.0).mean()
        loss = recons_loss + args.kl_weight * kl_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        current_step = step + resume_step

        if current_step % args.log_interval == 0:
            logger.logkv("step", current_step)
            logger.logkv("loss", loss.item())
            logger.logkv("recons_loss", recons_loss.item())
            logger.logkv("kl_loss", kl_loss.item())
            logger.dumpkvs()
            
        if current_step > 0 and current_step % args.save_interval == 0:
            if dist.get_rank() == 0:
                logger.log(f"saving model at step {current_step}...")
                model_path = os.path.join(logger.get_dir(), f"vae_model_{current_step:06d}.pt")
                opt_path = os.path.join(logger.get_dir(), f"opt_vae_{current_step:06d}.pt")
                th.save(ddp_vae.module.state_dict(), model_path)
                th.save(opt.state_dict(), opt_path)
            
            logger.log("saving training samples...")
            sample_and_save_vae(ddp_vae, args, current_step)

        step += 1

def create_argparser():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defaults = model_and_diffusion_defaults()
    defaults.pop('image_size', None)
    
    defaults.update(dict(
        image_size_h=256,
        image_size_w=160,
        lr=1e-4, 
        batch_size=32,
        lr_anneal_steps=300000, 
        save_dir=os.path.join(repo_root, "ckpt", "cgasf_ablation_vae_ablation_256x160"),
        latent_dim=256, 
        kl_weight=1e-6,
        data_dir=os.path.join(repo_root, "cgasf_ablation"),
        num_channels=128, 
        num_res_blocks=3, 
        attention_type='triple', 
        attention_resolutions="32,16,8",
        channel_mult="1,1,2,3,4", 
        use_scale_shift_norm=True,
        resume_checkpoint="", # 新增
        log_interval=10, # 从 diffusion 脚本中添加
        save_interval=5000, # 从 diffusion 脚本中添加
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

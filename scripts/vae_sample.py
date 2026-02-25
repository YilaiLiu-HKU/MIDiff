"""
Sample images from a trained VAE model.
Natively supports non-square images (e.g., 256x160).

** Corrected Version (v3) - Aligns model structure with vae_train.py **
"""

import argparse
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

# 复用您现有的工具
import dist_util, logger
from script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
# 导入 VAE 组件 (与 vae_train.py 完全一致)
from unet import EncoderUNetModel
from unet_new import DecoderUNetModel
from nn import timestep_embedding

# Reparameterize (仅在训练时需要，但我们保持类定义一致)
def reparameterize(mu, logvar):
    std = th.exp(0.5 * logvar)
    eps = th.randn_like(std)
    return mu + eps * std

class VAE(nn.Module):
    """
    VAE wrapper - ** MUST match the VAE class in vae_train.py EXACTLY **
    for state_dict loading.
    
    The forward method is modified for sampling (z -> x) instead of training (x -> x).
    """
    def __init__(self, args, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size_h = args.image_size_h
        self.image_size_w = args.image_size_w
        args.image_size=256
        encoder_args = args_to_dict(args, model_and_diffusion_defaults().keys())
        
        # 1. 初始化编码器 (Encoder)
        # ** 必须定义 self.encoder 才能加载 checkpoint **
        #
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
        #
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
        
        # 覆盖 latent_proj 层
        self.decoder.latent_proj = nn.Linear(
            latent_dim, 
            bottleneck_ch * self.bottleneck_res_h * self.bottleneck_res_w
        )
        
        # 4. Tanh 激活函数 (在训练脚本中添加)
        self.final_act = nn.Tanh()

    def forward(self, z):
        """
        ** 修改后的 Forward：仅用于采样 **
        输入 z (latent vector)，输出生成的图像。
        """
        
        t_zeros = th.zeros(z.shape[0], device=z.device, dtype=th.long)
        
        # 解码 (逻辑同 vae_train.py)
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
        recons = self.decoder.out(h_dec)
        
        return self.final_act(recons) # [B, 1, H, W]

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating VAE sampler model (non-square)...")
    model = VAE(args, args.latent_dim).to(dist_util.dev())
    
    logger.log(f"loading model from checkpoint: {args.model_path}...")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    
    ddp_model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
    )
    ddp_model.eval()

    logger.log("sampling...")
    all_images = []
    
    #
    while len(all_images) * args.batch_size * dist.get_world_size() < args.num_samples:
        # 动态计算当前 rank 需要生成的 batch size
        num_remaining = args.num_samples - len(all_images) * args.batch_size * dist.get_world_size()
        num_per_rank = (num_remaining + dist.get_world_size() - 1) // dist.get_world_size()
        current_batch_size = min(args.batch_size, num_per_rank)
        
        if current_batch_size <= 0:
             break

        # 从 Prior N(0, I) 采样 z
        z = th.randn(current_batch_size, args.latent_dim, device=dist_util.dev())
        
        with th.no_grad():
            sample = ddp_model.module(z) # [B, 1, 256, 160]

        # 反归一化到 [0, 255] uint8
        #
        sample = sample.permute(0, 2, 3, 1).contiguous() # [B, H, W, C]

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        
        # 只添加本 rank 实际生成的样本，避免多余的padding
        all_images.extend([s.cpu().numpy() for s in gathered_samples])
        logger.log(f"created {len(all_images) * dist.get_world_size()} samples (approx)")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"vae_samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

        # --- 新增：保存可视化PNG图像 ---
        #
        logger.log("saving individual sample images for visualization...")
        from PIL import Image # 导入 PIL
        
        # 创建一个子目录
        sampled_dir = os.path.join(logger.get_dir(), "sampled_images_vae")
        os.makedirs(sampled_dir, exist_ok=True)
        
        for i, img_np_hwc in enumerate(arr):
            # arr 已经是 [N, H, W, C] 且类型为 uint8
            # Squeeze a channel dim (C=1) to get [H, W]
            img_np_hw = np.squeeze(img_np_hwc, axis=-1)
            
            # 保存图像
            img_np_hw = ((img_np_hw + 1) * 127.5).clip(0, 255).astype(np.uint8)
            out_path_png = os.path.join(sampled_dir, f"sample_{i:06d}.png")
            Image.fromarray(img_np_hw, mode='L').save(out_path_png)
        
        logger.log(f"saved {len(arr)} individual samples to {sampled_dir}")
        # --- 可视化代码结束 ---

    dist.barrier()
    logger.log("VAE sampling complete")

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.pop('image_size', None)
    
    defaults.update(dict(
        image_size_h=256,
        image_size_w=160,
        num_samples=3000,
        batch_size=32, # 采样时可以使用更大的 batch_size
        save_dir="/data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr_vae_ablation_256x160",
        latent_dim=256, 
        model_path="", # VAE 模型 checkpoints 路径 (e.g., vae_model_XXXXXX.pt)
        # 保持与您的扩散模型指令一致
        num_channels=128, 
        num_res_blocks=3, 
        attention_type='triple', 
        attention_resolutions="32,16,8",
        channel_mult="1,1,2,3,4",
        use_scale_shift_norm=True,
    ))
    defaults.pop('lr', None)
    defaults.pop('lr_anneal_steps', None)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
"""
Sample images from a trained NVAE model.
Supports non-square images (256x160).
"""

import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from PIL import Image

import dist_util, logger
from script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from nvae_modules import NVAE


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
    
    while len(all_images) * args.batch_size * dist.get_world_size() < args.num_samples:
        num_remaining = args.num_samples - len(all_images) * args.batch_size * dist.get_world_size()
        num_per_rank = (num_remaining + dist.get_world_size() - 1) // dist.get_world_size()
        current_batch_size = min(args.batch_size, num_per_rank)
        
        if current_batch_size <= 0:
            break
        
        with th.no_grad():
            sample = ddp_model.module.sample(current_batch_size, dist_util.dev())
        
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        
        all_images.extend([s.cpu().numpy() for s in gathered_samples])
        logger.log(f"created {len(all_images) * dist.get_world_size()} samples (approx)")
    
    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"nvae_samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        
        logger.log("saving individual sample images...")
        sampled_dir = os.path.join(logger.get_dir(), "sampled_images_nvae")
        os.makedirs(sampled_dir, exist_ok=True)
        
        for i, img_np_hwc in enumerate(arr):
            img_np_hw = np.squeeze(img_np_hwc, axis=-1)
            out_path_png = os.path.join(sampled_dir, f"sample_{i:06d}.png")
            Image.fromarray(img_np_hw, mode='L').save(out_path_png)
        
        logger.log(f"saved {len(arr)} individual samples to {sampled_dir}")
    
    dist.barrier()
    logger.log("NVAE sampling complete")


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.pop('image_size', None)
    
    defaults.update(dict(
        image_size_h=256,
        image_size_w=160,
        num_samples=3000,
        batch_size=32,
        save_dir="/data/yilai/MiDiff/ckpt/ckpt/tiff_log_thr_nvae_ablation_256x160",
        latent_dim=256,
        base_channels=64,
        num_scales=3,
        num_cells_per_scale=2,
        model_path="",  # NVAE model checkpoint path
    ))
    
    for key in ['lr', 'lr_anneal_steps', 'num_channels', 'num_res_blocks',
                'attention_type', 'attention_resolutions', 'channel_mult',
                'use_scale_shift_norm', 'dropout', 'use_checkpoint',
                'num_heads', 'num_head_channels', 'use_fp16',
                'resblock_updown', 'use_new_attention_order']:
        defaults.pop(key, None)
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

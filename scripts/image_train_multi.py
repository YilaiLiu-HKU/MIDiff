"""
Train a diffusion model on images.
"""

import argparse
import os
os.chdir(r"/home/yilai/projects/poster/NetDiffus")

#import guided-difusion-0.0.0 as guided_diffusion
#import guided_diffusion
#from guided_diffusion import dist_util, logger
#from guided_diffusion.image_datasets import load_data
#from guided_diffusion.resample import create_named_schedule_sampler
#from guided_diffusion.script_util import (
#    model_and_diffusion_defaults,
#    create_model_and_diffusion,
#    args_to_dict,
#    add_dict_to_argparser,
#)
#from guided_diffusion.train_util import TrainLoop
import dist_util, logger
from image_datasets import load_data
from resample import create_named_schedule_sampler
from script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from train_util import TrainLoop
import torch

def main():
    raw_args = create_argparser().parse_args()

# 拆成列表
    data_dirs = [d.strip() for d in raw_args.data_dir.split(",") if d.strip()]
    save_dirs = [s.strip() for s in raw_args.save_dir.split(",") if s.strip()]

    if len(data_dirs) != len(save_dirs):
        raise ValueError("data_dir 与 save_dir 列表长度必须一致！")

    # 按顺序循环训练
    for data_dir, save_dir in zip(data_dirs, save_dirs):
        # 为本次循环构造一份独立 args
        args = argparse.Namespace(**vars(raw_args))
        args.data_dir = data_dir
        args.save_dir = save_dir



        dist_util.setup_dist()
        logger.configure(dir=args.save_dir)
        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        logger.log("creating data loader...")
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )

        # 保存预处理后的第一张图像
        import numpy as np
        import os
        import torch
        import torchvision.utils as vutils
        batch, mask, max_value,cond = next(data)
        first_img = batch[3]  # 取第一张
        #
        # 反归一化到0-255
        img = ((first_img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        # shape: (C, H, W) -> (H, W, C)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        # 保存为png
        from PIL import Image
        save_dir = args.save_dir if hasattr(args, 'save_dir') else '.'
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, 'preprocessed_first_image.png')
        print(img_np.shape)
        Image.fromarray(np.squeeze(img_np)).save(img_path)
        logger.log(f"Saved preprocessed first image to {img_path}")

        logger.log("training...")
        TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        special_weight=args.special_weight,
        cos_weight=args.cos_weight,
        using_MAE=args.using_MAE,
        post_traffic=args.post_traffic,
        use_heatmap=args.use_heatmap,
        use_FFT=args.use_FFT,
        discriminator= None,
        use_pixel_refiner=args.use_pixel_refiner    

    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=2e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        special_weight=1.0,  # 为有语义区域设定更高损失
        save_dir='128/iterate/df/synth_models',
        cos_weight=0.0,
        attention_type='origin',
        backbone_type='resnet',
        backbone='unet',
        using_MAE=False,
        post_traffic=False,
        use_heatmap=False,
        use_Norm=False,
        predict_xstart=False,
        use_FFT=False,
        use_discriminator=False,
        use_tile=False,
        use_pre_routing_bias=False,
        use_pixel_refiner=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

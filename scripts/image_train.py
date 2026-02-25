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
from moe_discriminator import Discriminator
import datetime
import torch.distributed as dist

def save_training_config(args, save_dir):
    """
    保存训练参数配置到txt文件
    
    Args:
        args: 参数对象
        save_dir: 保存目录
    """
    # 只在主进程中保存（如果已初始化分布式，则只在rank 0保存）
    try:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except:
        pass  # 如果dist未初始化，继续执行
        config_path = os.path.join(save_dir, "training_config.txt")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("训练参数配置\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # 按类别组织参数
            categories = {
                "数据相关": [
                    "data_dir", "batch_size", "microbatch", "image_size", 
                    "class_cond", "use_heatmap", "use_Norm", "use_tile"
                ],
                "模型相关": [
                    "backbone", "backbone_type", "attention_type", 
                    "num_channels", "num_res_blocks", "num_heads",
                    "channel_mult", "dropout", "use_checkpoint",
                    "use_scale_shift_norm", "resblock_updown"
                ],
                "扩散过程相关": [
                    "diffusion_steps", "noise_schedule", "timestep_respacing",
                    "learn_sigma", "use_kl", "predict_xstart",
                    "rescale_timesteps", "rescale_learned_sigmas"
                ],
                "训练相关": [
                    "lr", "weight_decay", "lr_anneal_steps", "schedule_sampler",
                    "ema_rate", "use_fp16", "fp16_scale_growth",
                    "log_interval", "save_interval", "resume_checkpoint"
                ],
                "损失函数相关": [
                    "special_weight", "cos_weight", "using_MAE",
                    "post_traffic", "use_FFT"
                ],
                "其他": [
                    "use_discriminator", "use_pixel_refiner", 
                    "use_pre_routing_bias", "save_dir"
                ]
            }
            
            # 获取所有参数
            all_params = vars(args)
            
            # 按类别写入
            for category, param_list in categories.items():
                f.write(f"\n【{category}】\n")
                f.write("-" * 80 + "\n")
                for param_name in param_list:
                    if hasattr(args, param_name):
                        value = getattr(args, param_name)
                        # 格式化输出
                        if isinstance(value, bool):
                            value_str = "True" if value else "False"
                        elif isinstance(value, (list, tuple)):
                            value_str = str(value)
                        else:
                            value_str = str(value)
                        f.write(f"  {param_name:30s} = {value_str}\n")
            
            # 写入未分类的参数
            written_params = set()
            for param_list in categories.values():
                written_params.update(param_list)
            
            remaining_params = {k: v for k, v in all_params.items() 
                              if k not in written_params}
            
            if remaining_params:
                f.write(f"\n【其他未分类参数】\n")
                f.write("-" * 80 + "\n")
                for param_name, value in sorted(remaining_params.items()):
                    if isinstance(value, bool):
                        value_str = "True" if value else "False"
                    elif isinstance(value, (list, tuple)):
                        value_str = str(value)
                    else:
                        value_str = str(value)
                    f.write(f"  {param_name:30s} = {value_str}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.log(f"训练参数配置已保存到: {config_path}")

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    
    # 保存训练参数配置
    save_training_config(args, logger.get_dir())
    
    logger.log("creating model and diffusion...")
    if args.use_discriminator:
        print("Using discriminator in diffusion model.")
        discriminator = Discriminator(channels=1)
        discriminator.to(dist_util.dev())
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
        use_heatmap=args.use_heatmap,
        use_Norm=args.use_Norm,
        use_tile=args.use_tile
    )

    # 保存预处理后的第一张图像
    import numpy as np
    import os
    import torch
    import torchvision.utils as vutils
    batch, mask, max_value,cond = next(data)
    first_img = batch[0]  # 取第一张


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
        discriminator=discriminator if args.use_discriminator else None,
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

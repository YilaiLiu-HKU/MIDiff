"""
消融实验脚本：测试TripletAttention在Transformer模型中的效果
"""

import argparse
import os
import json
import datetime
from pathlib import Path
import torch
import torch.distributed as dist

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


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=2e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        special_weight=1.0,
        save_dir='ablation_experiments',
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
        # 消融实验特定参数
        ablation_config='baseline',  # 'baseline', 'triplet_replace', 'triplet_add', 'hybrid'
        triplet_version='v1',  # 'v1' or 'v2'
        triplet_no_spatial=True,  # 是否使用空间注意力
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def save_experiment_config(args, save_dir, config_name):
    """保存实验配置"""
    if dist.get_rank() == 0:
        config_path = os.path.join(save_dir, f"{config_name}_config.json")
        config_dict = vars(args)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.log(f"实验配置已保存到: {config_path}")


def main():
    args = create_argparser().parse_args()
    
    # 根据消融配置设置模型参数
    ablation_configs = {
        'baseline': {
            'backbone': 'dit',
            'attention_type': 'origin',
            'description': '原始DiT模型，使用标准self-attention'
        },
        'triplet_replace': {
            'backbone': 'dit',
            'attention_type': 'triplet_replace',
            'description': 'DiT模型，用TripletAttention替换self-attention'
        },
        'triplet_add': {
            'backbone': 'dit',
            'attention_type': 'triplet_add',
            'description': 'DiT模型，在self-attention后额外添加TripletAttention'
        },
        'hybrid': {
            'backbone': 'dit',
            'attention_type': 'hybrid',
            'description': 'DiT模型，混合使用self-attention和TripletAttention'
        },
    }
    
    if args.ablation_config not in ablation_configs:
        raise ValueError(f"未知的消融配置: {args.ablation_config}. "
                        f"可选: {list(ablation_configs.keys())}")
    
    config = ablation_configs[args.ablation_config]
    args.backbone = config['backbone']
    args.attention_type = config['attention_type']
    
    # 设置保存目录
    experiment_name = f"{args.ablation_config}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_dir = os.path.join(args.save_dir, experiment_name)
    
    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    
    # 保存实验配置
    save_experiment_config(args, logger.get_dir(), args.ablation_config)
    
    logger.log(f"开始消融实验: {config['description']}")
    logger.log(f"配置: {args.ablation_config}")
    logger.log(f"TripletAttention版本: {args.triplet_version}")
    logger.log(f"TripletAttention使用空间注意力: {not args.triplet_no_spatial}")
    
    logger.log("creating model and diffusion...")
    
    # 获取所有参数
    model_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    # 添加triplet相关参数（这些参数不在defaults中，需要手动添加）
    model_kwargs['triplet_version'] = getattr(args, 'triplet_version', 'v1')
    model_kwargs['triplet_no_spatial'] = getattr(args, 'triplet_no_spatial', True)
    
    # 创建模型
    model, diffusion = create_model_and_diffusion(**model_kwargs)
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
        discriminator=None,
        use_pixel_refiner=args.use_pixel_refiner
    ).run_loop()


if __name__ == "__main__":
    main()

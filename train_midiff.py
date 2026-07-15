"""Train MIDiff on C-GASF images."""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
os.chdir(REPO_ROOT)

import dist_util, logger
from image_datasets import load_data
from resample import create_named_schedule_sampler
from script_util import (
    midiff_model_and_diffusion_defaults,
    create_midiff_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from train_util import TrainLoop
import torch
import datetime
import torch.distributed as dist

def save_training_config(args, save_dir):
    """Save the MIDiff training configuration on rank 0."""
    try:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except:
        pass  # 如果dist未初始化，继续执行
    config_path = os.path.join(save_dir, "training_config.txt")

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MIDiff training configuration\n")
        f.write("=" * 80 + "\n")
        f.write(f"created_at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        categories = {
            "Data": [
                "data_dir", "batch_size", "microbatch", "image_size",
                "class_cond"
            ],
            "MIDiff model": [
                "attention_type", "num_channels", "num_res_blocks", "num_heads",
                "channel_mult", "dropout", "use_checkpoint",
                "use_scale_shift_norm", "resblock_updown"
            ],
            "Diffusion": [
                "diffusion_steps", "noise_schedule", "timestep_respacing",
                "learn_sigma", "use_kl", "predict_xstart",
                "rescale_timesteps", "rescale_learned_sigmas"
            ],
            "Training": [
                "lr", "weight_decay", "lr_anneal_steps", "schedule_sampler",
                "ema_rate", "use_fp16", "fp16_scale_growth",
                "log_interval", "save_interval", "resume_checkpoint"
            ],
            "Loss": [
                "special_weight"
            ],
            "Output": [
                "save_dir"
            ]
        }

        all_params = vars(args)

        for category, param_list in categories.items():
            f.write(f"\n[{category}]\n")
            f.write("-" * 80 + "\n")
            for param_name in param_list:
                if hasattr(args, param_name):
                    value = getattr(args, param_name)
                    if isinstance(value, bool):
                        value_str = "True" if value else "False"
                    elif isinstance(value, (list, tuple)):
                        value_str = str(value)
                    else:
                        value_str = str(value)
                    f.write(f"  {param_name:30s} = {value_str}\n")

        written_params = set()
        for param_list in categories.values():
            written_params.update(param_list)

        remaining_params = {k: v for k, v in all_params.items()
                          if k not in written_params}

        if remaining_params:
            f.write(f"\n[Other]\n")
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

    logger.log(f"MIDiff training configuration saved to: {config_path}")

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    save_training_config(args, logger.get_dir())

    logger.log("creating MIDiff model and diffusion...")
    model, diffusion = create_midiff_model_and_diffusion(
        **args_to_dict(args, midiff_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating C-GASF data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    batch, mask, max_value,cond = next(data)
    first_img = batch[0]


    logger.log("training MIDiff...")

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

    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir=os.path.join(REPO_ROOT, "cgasf"),
        schedule_sampler="uniform",
        lr=5e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        special_weight=1.0,
        save_dir=os.path.join(REPO_ROOT, "ckpt", "midiff"),
        attention_type='triple',
        predict_xstart=False,
    )
    model_defaults = midiff_model_and_diffusion_defaults()
    model_defaults.update(defaults)
    defaults = model_defaults
    parser = argparse.ArgumentParser(description="Train MIDiff on C-GASF images.")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

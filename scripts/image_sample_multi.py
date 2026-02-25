"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import csv

import numpy as np
import torch as th
import torch.distributed as dist

import dist_util, logger
from script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def center_crop_192_140(image,image_size_second=140):
    """将图像中心裁剪到288*140"""

    _,h, w,c = image.shape
    target_h, target_w = 192, image_size_second
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return image[:,top:top+target_h, left:left+target_w]
def main():
    raw_args = create_argparser().parse_args()

    # 拆解成列表
    model_paths = [p.strip() for p in raw_args.model_path.split(",") if p.strip()]
    save_dirs   = [s.strip() for s in raw_args.save_dir.split(",")   if s.strip()]
    image_size_seconds   = [int(i.strip()) for i in raw_args.image_size_second.split(",")   if i.strip()]
    print(image_size_seconds)
    if len(model_paths) != len(save_dirs):
        raise ValueError("model_path 与 save_dir 列表长度必须一致！")

    # 依次采样
    for model_path, save_dir,image_size_second in zip(model_paths, save_dirs,image_size_seconds):
        # 为本次循环构造独立 args
        args = argparse.Namespace(**vars(raw_args))
        args.model_path = model_path
        args.save_dir   = save_dir
        args.image_size_second=image_size_second
        dist_util.setup_dist()
        logger.configure(dir=args.save_dir)

        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu"),strict=False
        )
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        print(logger.get_dir())
        logger.log("sampling...")
        all_images = []
        all_labels = []
        all_max_predictions = []
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            #import pdb;pdb.set_trace()
            ###这里为了不规整的采样图像添加
            print(f"到这里了{dist.get_rank()}")
 
            if args.image_size_second==0:
                image_size_second=args.image_size
            else:
                image_size_second=args.image_size_second
            if image_size_second<=160:
                sample_size_second=160
            else:
                sample_size_second=224
            sample,max_prediction = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, sample_size_second),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            print(f"采样完了{dist.get_rank()}")
            ###for viridis
            #sample = (((sample + 1) * 126).clamp(0, 252)+1).to(th.uint8)
            #sample = (((sample+1) * 127.5).clamp(0, 255)).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            #import ;.set_trace
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            #gathered_max_predictions = [th.zeros_like(max_prediction) for _ in range(dist.get_world_size())]
            print(f"到这里了{dist.get_rank()}")
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            print(f"gather完了{dist.get_rank()}")
            #dist.all_gather(gathered_max_predictions, max_prediction)  # gather max predictions
            #import pdb;pdb.set_trace()
            sample=center_crop_192_140(sample,image_size_second)
            print(f"裁剪完了{dist.get_rank()}")
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            #all_max_predictions.extend([max_pred.cpu().numpy() for max_pred in gathered_max_predictions])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                print(f"到这里了{dist.get_rank()}")
                dist.all_gather(gathered_labels, classes)
                print(f"gather完了{dist.get_rank()}")
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        
        #max_predictions_arr = np.concatenate(all_max_predictions, axis=0)
        #max_predictions_arr = max_predictions_arr[: args.num_samples]
        def center_crop(image, target_h=288, target_w=340):
            if image.ndim == 2:
                h, w = image.shape
                top = (h - target_h) // 2
                left = (w - target_w) // 2
                return image[top:top+target_h, left:left+target_w]
            else:
                raise ValueError("Unsupported shape for cropping")

        # 对所有样本裁剪
        #arr = np.array([center_crop(sample) for sample in arr])
        ##### 这里为了忽略填充的部分，进行一个裁剪
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
                    
            import os
            out_path = os.path.join(logger.get_dir(), f"{args.model_path.split('/')[-1]}_samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")

        # ====== FID评估 ======
        if dist.get_rank() == 0:
            import os
            from PIL import Image
            try:
                from pytorch_fid import fid_score
            except ImportError:
                print("请先 pip install pytorch-fid")
                return
            sampled_dir = os.path.join(args.save_dir, "sampled_images")
            os.makedirs(sampled_dir, exist_ok=True)
            
            # 保存最大值预测到CSV文件
            """csv_path = os.path.join(args.save_dir, f"{args.model_path.split('/')[-1]}_max_predictions.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['image_index', 'max_prediction'])  # 写入表头
                for i, max_pred in enumerate(max_predictions_arr):
                    writer.writerow([i, float(max_pred)])
            logger.log(f"保存最大值预测到: {csv_path}")"""
            
            # 中心裁剪函数

            
            # 保存裁剪后的图像用于FID计算
            for i, img in enumerate(arr):

                img_uint8 = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)

            # 2. Squeeze 掉单通道维度
                img_squeezed = np.squeeze(img_uint8)
                save_path = os.path.join(sampled_dir, f"{i:05d}.png")
                Image.fromarray(img_squeezed, mode='L').save(save_path)
            real_dir = "/home/yilai/poster/NetDiffus/dataset_tiff_visual"
            # 处理真实图片：中心裁剪到288*140

            # 计算FID（使用裁剪后的图片）
            """fid_value = fid_score.calculate_fid_given_paths(
                [real_dir, sampled_dir],
                batch_size=args.num_samples,
                device=th.device("cuda" if th.cuda.is_available() else "cpu"),
                dims=2048,
            )
            print(f"FID: {fid_value}")
            logger.log(f"FID: {fid_value}")"""


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=5,
        batch_size=64,
        use_ddim=False,
        model_path="",
        save_dir='128/iterate/df/synth_models',
        image_size_second='',
        attention_type='origin',
       
        backbone_type='resnet',
        backbone='unet',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

import copy
import functools
import os
import math
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from PIL import Image
#from . import dist_util, logger
#from .fp16_util import MixedPrecisionTrainer
#from .nn import update_ema
#from .resample import LossAwareSampler, UniformSampler
import dist_util, logger
from fp16_util import MixedPrecisionTrainer
from nn import update_ema
from resample import LossAwareSampler, UniformSampler
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
import torch
matplotlib.use('Agg')  # 使用非交互式后端
torch.autograd.set_detect_anomaly(True)
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        special_weight=1.0,
        cos_weight=0.0,
        using_MAE=False,
        post_traffic=False,
        use_heatmap=False,
        use_FFT=False,
        discriminator=None,
        use_pixel_refiner=False
    ):
        self.model = model
        self.discriminator = discriminator
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.use_pixel_refiner = use_pixel_refiner
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.special_weight = special_weight
        self.cos_weight=cos_weight
        self.using_MAE=using_MAE
        self.post_traffic=post_traffic
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.use_heatmap = use_heatmap
        self.use_FFT=use_FFT
        # 最优模型相关变量
        self.best_loss = float('inf')
        self.last_best_update_step = 0
        # 损失监控相关
        self.loss_history = {
            'step': [],
            'loss': [],
            'mse': [],
            'vb': [],
            'aux_loss': [],
            'cos': [],
            'traffic_mse': []
        }
        self.plot_interval = 100  # 每100步绘制一次
        self.loss_save_interval = 50  # 每50步保存一次损失到CSV

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        if self.use_pixel_refiner:
            for name, param in self.model.named_parameters():
                if "pixel_refiner" not in name:
                    param.requires_grad = False
            
            # 筛选出需要训练的参数
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            
            self.opt = AdamW(
                trainable_params, lr=self.lr, weight_decay=self.weight_decay
            )
            logger.log("Optimizer configured to train only the PixelRefiner.")
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
            )
        if self.discriminator is not None:
            self.discriminator_trainer = MixedPrecisionTrainer(
                model=self.discriminator,
                use_fp16=self.use_fp16,
                fp16_scale_growth=fp16_scale_growth,
            )
            self.discriminator_opt = AdamW(
        self.discriminator_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
    )
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True ,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    ),strict=False
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        """if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)"""

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if not main_checkpoint:  # No checkpoint to resume from
            return
            
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        
        if not bf.exists(opt_checkpoint):
            logger.log(f"No optimizer state found at {opt_checkpoint}")
            return
        
        try:
            logger.log(f"Attempting to load optimizer state from {opt_checkpoint}")
            
            # Attempt to load with retries
            for attempt in range(3):  # Try up to 3 times
                try:
                    state_dict = dist_util.load_state_dict(
                        opt_checkpoint, 
                        map_location=dist_util.dev()
                    )
                    
                    # Verify the loaded state contains required keys
                    if not isinstance(state_dict, dict) or 'state' not in state_dict:
                        raise ValueError("Invalid optimizer state format")
                    
                    self.opt.load_state_dict(state_dict)
                    logger.log("Optimizer state loaded successfully")
                    return  # Success - exit the function
                    
                except Exception as e:
                    if attempt == 2:  # Last attempt failed
                        logger.log(f"Failed to load optimizer state after {attempt+1} attempts: {e}")
                    else:
                        logger.log(f"Attempt {attempt+1} failed, retrying...")
                        import time
                        time.sleep(1)  # Wait before retrying
            
            logger.log("Will continue with fresh optimizer state")
            
        except Exception as e:
            logger.log(f"Unexpected error loading optimizer state: {e}")
            logger.log("Will continue with fresh optimizer state")

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            
            batch, mask,max_value,cond = next(self.data)
            self.run_step(batch, cond,mask,max_value)
            if self.step % self.log_interval == 0:
                print(self.step+self.resume_step)
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            # 每10步检查是否需要保存最优模型
            """if self.step % 100 == 0 and len(self.loss_history['loss']) > 0:
                current_loss = self.loss_history['loss'][-1]
                self.save_best_model(current_loss)"""
            # Run for a finite amount of time in integration tests.
            if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond,mask=None,max_value=None):
        self.forward_backward(batch, cond,mask,max_value)
        took_step = self.mp_trainer.optimize(self.opt)
        self.discriminator_trainer.optimize(self.discriminator_opt) if self.discriminator is not None else None
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond,mask=None,max_value=None):
        self.mp_trainer.zero_grad()
        if self.discriminator is not None:
            self.discriminator_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            if mask is not None:
                micro_cond["padding_mask"]=mask
            # 传递special_weight参数
                micro_cond['post_traffic']=self.post_traffic
            if hasattr(self, 'special_weight'):
                micro_cond['special_weight'] = self.special_weight
                micro_cond['special_value']=-1,
                micro_cond['cos_weight']=self.cos_weight
                micro_cond['max_value']=max_value
                micro_cond['using_MAE']=self.using_MAE
                micro_cond['use_heatmap']=self.use_heatmap
                micro_cond['use_FFT']=self.use_FFT
                micro_cond['discriminator']=self.discriminator
                micro_cond['use_pixel_refiner']=self.use_pixel_refiner
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            # 收集损失数据用于监控
            if dist.get_rank() == 0:
                current_step = self.step + self.resume_step
                if len(self.loss_history['step']) == 0 or self.loss_history['step'][-1] != current_step:
                    self.loss_history['step'].append(current_step)
                    self.loss_history['loss'].append(loss.item())
                    self.loss_history['mse'].append(losses.get('mse', th.tensor(0.0)).mean().item())
            
                    self.loss_history['aux_loss'].append(losses.get('aux_loss', th.tensor(0.0)).item())
                    self.loss_history['cos'].append(losses.get('cos', th.tensor(0.0)).item())
                    self.loss_history['traffic_mse'].append(losses.get('traffic_mse', th.tensor(0.0)).item())
            


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
            
    def save_best_model(self, current_loss):
        """Save model when current loss is better than previous best"""
        if current_loss < 0.8 * self.best_loss:
            self.best_loss = current_loss
            
            # 保存模型参数
            if dist.get_rank() == 0:
                logger.log(f"Saving best model with loss: {current_loss}...")
                # 保存主模型
                state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
                with bf.BlobFile(bf.join(get_blob_logdir(), "model_best.pt"), "wb") as f:
                    th.save(state_dict, f)
                
                # 保存EMA模型
                for rate, params in zip(self.ema_rate, self.ema_params):
                    state_dict = self.mp_trainer.master_params_to_state_dict(params)
                    with bf.BlobFile(bf.join(get_blob_logdir(), f"ema_{rate}_best.pt"), "wb") as f:
                        th.save(state_dict, f)
                logger.log("Best model saved successfully!")

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        
        # 定期保存损失到CSV和绘制损失曲线
        if dist.get_rank() == 0:
            current_step = self.step + self.resume_step
            
            
            # 定期绘制损失曲线
            if current_step % self.save_interval == 0:
                self._plot_loss_curves()

    def save(self, current_loss=None):
        def save_checkpoint(rate, params, is_best=False):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                
                # 如果是最优模型，额外保存一份
                if is_best:
                    best_filename = "model_best.pt" if not rate else f"ema_{rate}_best.pt"
                    logger.log(f"Saving best model {rate}...")
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
                
                if is_best:
                    with bf.BlobFile(bf.join(get_blob_logdir(), best_filename), "wb") as f:
                        th.save(state_dict, f)

        # 检查是否是最优模型
        is_best = False
        """if current_loss is not None:
            current_step = self.step + self.resume_step
            # 当前损失小于历史最优损失的80%且距离上次更新至少有1000步
            if current_loss < 0.8 * self.best_loss and current_step - self.last_best_update_step >= 1000:
                self.best_loss = current_loss
                self.last_best_update_step = current_step
                is_best = True
                logger.log(f"New best model with loss: {current_loss}")"""
        
        save_checkpoint(0, self.mp_trainer.master_params, is_best)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params, is_best)
           
        if dist.get_rank() == 0:
            if self.discriminator is not None:
                logger.log(f"saving discriminator...")
                discriminator_state_dict = self.discriminator.state_dict()
                discriminator_filename = f"discriminator_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), discriminator_filename), "wb") as f:
                    th.save(discriminator_state_dict, f)
                logger.log(f"Discriminator saved to {discriminator_filename}")
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # --------- 新增采样保存逻辑 ---------
        def sample_and_save(model, suffix):
            model.eval()
            batch_size = 5  # 改为每次采样5张
            image_size = getattr(self, 'image_size', 256)
            class_cond = getattr(self, 'class_cond', False)
            model_kwargs = {}
            if class_cond:
                from script_util import NUM_CLASSES
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = self.diffusion.p_sample_loop
            second_image_size = 160
            sample,_ = sample_fn(
                model,
                (batch_size, 1, image_size, second_image_size),
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )
            #import pdb;pdb.set_trace()
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images = [sample.cpu().numpy() for sample in gathered_samples]
            arr = np.concatenate(all_images, axis=0)
            arr = arr[:batch_size]
            
            if dist.get_rank() == 0:
                import matplotlib.pyplot as plt
                height_crop = (256 - 192) // 2      # 高度方向裁剪16像素（上下各8）
                width_crop = (160 - 140) // 2        # 宽度方向裁剪20像素（左右各10）
                
                # 创建输出目录
                os.makedirs(os.path.join(logger.get_dir(), "samples"), exist_ok=True)
                
                for i in range(batch_size):
                    img_np = arr[i]
                    # 对称裁剪
                    img_cropped = img_np[
                        height_crop : height_crop + 192,  # 高度范围
                        width_crop : width_crop + 140,    # 宽度范围
                        :                                 # 保留所有通道
                    ]

                    out_path = os.path.join(
                        logger.get_dir(), 
                        "samples",
                        f"sample_{suffix}_{(self.step+self.resume_step):06d}_{i:02d}.png"
                    )
                    cropped_out_path = os.path.join(
                        logger.get_dir(),
                        "samples",
                        f"cropped_sample_{suffix}_{(self.step+self.resume_step):06d}_{i:02d}.png"
                    )
                    img_np=np.squeeze(img_np)
                    img_cropped=np.squeeze(img_cropped)
                    Image.fromarray(img_np, mode='L').save(out_path)
                    Image.fromarray( img_cropped, mode='L').save(cropped_out_path)
                
                logger.log(f"saving sample image to {out_path}")
            dist.barrier()

        # 保存当前模型采样
        sample_and_save(self.model, "model")
        # 保存ema模型采样（如有多个ema_rate则分别采样）
        for rate, params in zip(self.ema_rate, self.ema_params):
            # 创建ema模型副本并加载参数
            ema_model = copy.deepcopy(self.model)
            ema_model.load_state_dict(self.mp_trainer.master_params_to_state_dict(params),strict=False)
            
            sample_and_save(ema_model, f"ema_{rate}")
        # --------- 采样保存逻辑结束 ---------

        dist.barrier()

    def _save_loss_to_csv(self):
        """保存损失数据到CSV文件"""
        if dist.get_rank() == 0:
            csv_path = os.path.join(logger.get_dir(), "loss_history.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 写入表头
                writer.writerow(['step', 'loss', 'mse', 'vb', 'aux_loss', 'cos'])
                # 写入数据
                for i in range(len(self.loss_history['step'])):
                    writer.writerow([
                        self.loss_history['step'][i],
                        self.loss_history['loss'][i],
                        self.loss_history['mse'][i],
                        self.loss_history['vb'][i],
                        self.loss_history['aux_loss'][i],
                        self.loss_history['cos'][i]
                    ])
            logger.log(f"损失数据已保存到: {csv_path}")

    def _plot_loss_curves(self):
        """绘制损失曲线，只显示最近10000个step"""
        if dist.get_rank() == 0 and len(self.loss_history['step']) > 0:
            # 获取最近10000个数据点
            window_size = 2000
            start_idx = max(0, len(self.loss_history['step']) - window_size)
            
            # 获取要绘制的数据片段
            steps = self.loss_history['step'][start_idx:]
            losses = self.loss_history['loss'][start_idx:]
            mse_losses = self.loss_history['mse'][start_idx:]
            aux_losses = self.loss_history['traffic_mse'][start_idx:]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('训练损失曲线 (最近10000步)', fontsize=16)
            
            # 总损失
            axes[0, 0].plot(steps, losses, 'b-', label='Total Loss')
            axes[0, 0].set_title('总损失')
            axes[0, 0].set_xlabel('步数')
            axes[0, 0].set_ylabel('损失值')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # MSE损失
            axes[0, 1].plot(steps, mse_losses, 'r-', label='MSE Loss')
            axes[0, 1].set_title('MSE损失')
            axes[0, 1].set_xlabel('步数')
            axes[0, 1].set_ylabel('损失值')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # VB损失
            axes[1, 0].plot(steps, aux_losses, 'g-', label='VB Loss')
            axes[1, 0].set_title('aux')
            axes[1, 0].set_xlabel('步数')
            axes[1, 0].set_ylabel('损失值')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 辅助损失
            axes[1, 1].plot(steps, aux_losses, 'm-', label='Traffic Loss')
            axes[1, 1].set_title('流量损失')
            axes[1, 1].set_xlabel('步数')
            axes[1, 1].set_ylabel('损失值')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = os.path.join(logger.get_dir(), f"loss_curves_{self.step + self.resume_step:06d}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.log(f"损失曲线已保存到: {plot_path}")


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

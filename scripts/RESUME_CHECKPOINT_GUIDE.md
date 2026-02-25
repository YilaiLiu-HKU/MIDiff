# StyleGAN2 & NVAE 训练恢复指南

## 🔄 Resume Checkpoint 使用方法

### StyleGAN2

#### 1. 从 checkpoint 恢复训练

```bash
# 基础用法
CUDA_VISIBLE_DEVICES=5 python scripts/stylegan2_train.py \
    --data_dir /home/yilai/projects/poster/NetDiffus/tiff_log \
    --save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_ablation_256x160 \
    --batch_size 16 \
    --resume_checkpoint_g /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_ablation_256x160/stylegan2_G_010625.pt \
    --resume_checkpoint_d /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_ablation_256x160/stylegan2_D_010625.pt
```

#### 2. 只恢复 Generator（迁移学习）

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/stylegan2_train.py \
    --data_dir /home/yilai/projects/poster/NetDiffus/tiff_log \
    --save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_new \
    --batch_size 16 \
    --resume_checkpoint_g /path/to/stylegan2_G_010625.pt
```

#### 3. 文件结构
训练时会自动保存：
```
save_dir/
├── stylegan2_G_010625.pt          # Generator 权重
├── stylegan2_D_010625.pt          # Discriminator 权重
├── opt_G_010625.pt                # Generator 优化器状态
├── opt_D_010625.pt                # Discriminator 优化器状态
└── samples_stylegan2_during_train/
    ├── sample_010625_00.png
    ├── sample_010625_01.png
    └── ...
```

---

### NVAE

#### 1. 从 checkpoint 恢复训练

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/nvae_train.py \
    --data_dir /home/yilai/projects/poster/NetDiffus/tiff_log \
    --save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_nvae_ablation_256x160 \
    --batch_size 16 \
    --resume_checkpoint /data/yilai/MiDiff/ckpt/ckpt/tiff_log_nvae_ablation_256x160/nvae_model_005000.pt
```

#### 2. 文件结构
训练时会自动保存：
```
save_dir/
├── nvae_model_005000.pt           # NVAE 模型权重
├── opt_nvae_005000.pt             # 优化器状态
└── samples_nvae_during_train/
    ├── sample_005000_00.png
    ├── sample_005000_01.png
    └── ...
```

---

## 📋 完整命令示例

### 场景 1: 从头开始训练

#### StyleGAN2
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/stylegan2_train.py \
    --data_dir /home/yilai/projects/poster/NetDiffus/tiff_log \
    --save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_256x160 \
    --batch_size 32 \
    --lr_g 2e-3 \
    --lr_d 2e-3 \
    --lr_anneal_steps 300000
```

#### NVAE
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/nvae_train.py \
    --data_dir /home/yilai/projects/poster/NetDiffus/tiff_log \
    --save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_nvae_256x160 \
    --batch_size 32 \
    --lr 1e-3 \
    --lr_anneal_steps 300000
```

---

### 场景 2: 训练中断后恢复

#### StyleGAN2（假设在 step 10625 时中断）
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/stylegan2_train.py \
    --data_dir /home/yilai/projects/poster/NetDiffus/tiff_log \
    --save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_256x160 \
    --batch_size 32 \
    --resume_checkpoint_g /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_256x160/stylegan2_G_010625.pt \
    --resume_checkpoint_d /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_256x160/stylegan2_D_010625.pt \
    --lr_anneal_steps 300000
```

#### NVAE（假设在 step 5000 时中断）
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/nvae_train.py \
    --data_dir /home/yilai/projects/poster/NetDiffus/tiff_log \
    --save_dir /data/yilai/MiDiff/ckpt/ckpt/tiff_log_nvae_256x160 \
    --batch_size 32 \
    --resume_checkpoint /data/yilai/MiDiff/ckpt/ckpt/tiff_log_nvae_256x160/nvae_model_005000.pt \
    --lr_anneal_steps 300000
```

**注意**: 
- ✅ Step 计数会自动从 checkpoint 文件名中解析
- ✅ 优化器状态会自动加载（如果存在）
- ✅ 训练会从 `resume_step + 1` 继续

---

### 场景 3: 使用预训练模型微调（新数据集）

#### StyleGAN2
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/stylegan2_train.py \
    --data_dir /home/yilai/projects/poster/NetDiffus/NEW_DATASET \
    --save_dir /data/yilai/MiDiff/ckpt/ckpt/new_dataset_stylegan2 \
    --batch_size 16 \
    --lr_g 1e-4 \
    --lr_d 1e-4 \
    --resume_checkpoint_g /path/to/pretrained/stylegan2_G_300000.pt \
    --resume_checkpoint_d /path/to/pretrained/stylegan2_D_300000.pt \
    --lr_anneal_steps 50000
```
**提示**: 使用较小的学习率进行微调

---

## 🔍 如何找到最新的 checkpoint

### 方法 1: 手动查找
```bash
ls -lt /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_256x160/*.pt | head -2
```

### 方法 2: 使用脚本自动查找（创建辅助脚本）
```bash
# 创建 find_latest_checkpoint.sh
cat > find_latest_checkpoint.sh << 'EOF'
#!/bin/bash
DIR=$1
MODEL_TYPE=$2  # "stylegan2_G" or "nvae_model"

if [ -z "$DIR" ] || [ -z "$MODEL_TYPE" ]; then
    echo "Usage: $0 <checkpoint_dir> <model_type>"
    echo "Example: $0 /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_256x160 stylegan2_G"
    exit 1
fi

LATEST=$(ls -t ${DIR}/${MODEL_TYPE}_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
    echo "No checkpoint found for ${MODEL_TYPE} in ${DIR}"
    exit 1
fi

echo "Latest checkpoint: ${LATEST}"
echo ""
echo "To resume training, use:"
if [[ "$MODEL_TYPE" == "stylegan2_G" ]]; then
    STEP=$(basename $LATEST | sed 's/stylegan2_G_\(.*\)\.pt/\1/')
    D_CKPT="${DIR}/stylegan2_D_${STEP}.pt"
    echo "--resume_checkpoint_g ${LATEST}"
    echo "--resume_checkpoint_d ${D_CKPT}"
else
    echo "--resume_checkpoint ${LATEST}"
fi
EOF

chmod +x find_latest_checkpoint.sh
```

**使用示例**:
```bash
# StyleGAN2
./find_latest_checkpoint.sh /data/yilai/MiDiff/ckpt/ckpt/tiff_log_stylegan2_256x160 stylegan2_G

# NVAE
./find_latest_checkpoint.sh /data/yilai/MiDiff/ckpt/ckpt/tiff_log_nvae_256x160 nvae_model
```

---

## ⚠️ 注意事项

### 1. 优化器状态加载
- ✅ **自动加载**: 如果 `opt_G_NNNNNN.pt` 和 `opt_D_NNNNNN.pt` 存在，会自动加载
- ⚠️ **如果不存在**: 会使用新的优化器状态，从头开始累积动量（可能导致短暂的训练不稳定）

### 2. Step 计数
- ✅ 会从 checkpoint 文件名自动解析（例如 `stylegan2_G_010625.pt` → step 10625）
- ✅ 训练日志中的 step 会正确显示累积值
- ✅ 保存的新 checkpoint 会使用正确的 step 编号

### 3. 兼容性
- ⚠️ 确保 checkpoint 的模型架构参数一致：
  - `latent_dim`, `w_dim`, `image_size_h`, `image_size_w` 等
- ⚠️ 如果参数不匹配，会报错

### 4. 多 GPU 训练
```bash
# StyleGAN2 (4 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 mpiexec -n 4 python scripts/stylegan2_train.py \
    --resume_checkpoint_g /path/to/stylegan2_G_010625.pt \
    --resume_checkpoint_d /path/to/stylegan2_D_010625.pt \
    ...

# NVAE (4 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 mpiexec -n 4 python scripts/nvae_train.py \
    --resume_checkpoint /path/to/nvae_model_005000.pt \
    ...
```

---

## 🎯 快速参考

| 任务 | StyleGAN2 参数 | NVAE 参数 |
|------|---------------|-----------|
| 从头训练 | *无需参数* | *无需参数* |
| 恢复训练 | `--resume_checkpoint_g` + `--resume_checkpoint_d` | `--resume_checkpoint` |
| 仅恢复生成器 | `--resume_checkpoint_g` | `--resume_checkpoint` |
| 仅恢复判别器 | `--resume_checkpoint_d` | N/A |

---

## 📞 常见问题

**Q: 如何验证 checkpoint 是否正确加载？**
A: 查看训练日志开头：
```
loading Generator from checkpoint: /path/to/stylegan2_G_010625.pt...
loading Discriminator from checkpoint: /path/to/stylegan2_D_010625.pt...
loading Generator optimizer state from checkpoint: /path/to/opt_G_010625.pt...
loading Discriminator optimizer state from checkpoint: /path/to/opt_D_010625.pt...
```

**Q: 可以只恢复模型权重，不恢复优化器状态吗？**
A: 可以！删除或重命名 `opt_*.pt` 文件即可，程序会自动跳过并使用新的优化器。

**Q: 恢复训练后 loss 突然变化正常吗？**
A: 
- ✅ 如果优化器状态正确加载，loss 应该平滑过渡
- ⚠️ 如果优化器状态缺失，可能有短暂波动（1-2k steps 后恢复正常）

**Q: 可以从不同 batch_size 的 checkpoint 恢复吗？**
A: 可以，但优化器状态中的学习率调度可能不一致，建议删除优化器 checkpoint。

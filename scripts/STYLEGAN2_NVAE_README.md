# StyleGAN2 和 NVAE 无条件生成模型

## 概述

这是针对消融实验的高性能无条件生成模型实现，包括：

1. **StyleGAN2**: 最先进的 GAN 架构，具有自适应实例归一化 (AdaIN)、渐进式生长和高质量图像生成能力
2. **NVAE**: 具有多尺度层次潜变量的深度变分自编码器，在 VAE 中达到 SOTA 性能

两个模型都原生支持**非正方形图像** (256×160)。

---

## 文件说明

### 核心模块
- `stylegan2_modules.py`: StyleGAN2 的核心组件
  - `MappingNetwork`: Z → W 映射网络
  - `SynthesisNetwork`: W → Image 合成网络
  - `StyleGAN2Generator`: 完整生成器
  - `StyleGAN2Discriminator`: 判别器
  - `ModulatedConv2d`: 调制卷积层

- `nvae_modules.py`: NVAE 的核心组件
  - `NVAEEncoder`: 多尺度编码器
  - `NVAEDecoder`: 多尺度解码器
  - `NVAE`: 完整 VAE 模型
  - `ResidualCell`: 残差单元

### 训练脚本
- `stylegan2_train.py`: StyleGAN2 训练
- `nvae_train.py`: NVAE 训练

### 采样脚本
- `stylegan2_sample.py`: StyleGAN2 采样
- `nvae_sample.py`: NVAE 采样

---

## 使用方法

### 1. StyleGAN2 训练

```bash
# 单 GPU
CUDA_VISIBLE_DEVICES=0 python stylegan2_train.py \
    --data_dir /path/to/data \
    --save_dir /path/to/save \
    --batch_size 16 \
    --lr_g 2e-3 \
    --lr_d 2e-3 \
    --latent_dim 256 \
    --w_dim 512

# 多 GPU (推荐)
mpiexec -n 4 python stylegan2_train.py \
    --data_dir /path/to/data \
    --save_dir /path/to/save \
    --batch_size 16
```

**关键参数**:
- `--latent_dim`: Z 空间维度 (默认 256)
- `--w_dim`: W 空间维度 (默认 512)
- `--gp_lambda`: R1 梯度惩罚系数 (默认 10.0)
- `--d_steps`: 每个 G step 训练 D 的次数 (默认 1)

### 2. NVAE 训练

```bash
# 单 GPU
CUDA_VISIBLE_DEVICES=0 python nvae_train.py \
    --data_dir /path/to/data \
    --save_dir /path/to/save \
    --batch_size 16 \
    --lr 1e-3 \
    --latent_dim 256 \
    --num_scales 3

# 多 GPU (推荐)
mpiexec -n 4 python nvae_train.py \
    --data_dir /path/to/data \
    --save_dir /path/to/save \
    --batch_size 16
```

**关键参数**:
- `--latent_dim`: 潜变量维度 (默认 256)
- `--num_scales`: 多尺度层数 (默认 3)
- `--base_channels`: 基础通道数 (默认 64)
- `--kl_weight`: KL 散度权重 (默认 1e-4)
- `--grad_clip`: 梯度裁剪阈值 (默认 1.0)

### 3. StyleGAN2 采样

```bash
mpiexec -n 4 python stylegan2_sample.py \
    --model_path /path/to/stylegan2_G_XXXXXX.pt \
    --save_dir /path/to/save \
    --num_samples 3000 \
    --batch_size 32
```

### 4. NVAE 采样

```bash
mpiexec -n 4 python nvae_sample.py \
    --model_path /path/to/nvae_model_XXXXXX.pt \
    --save_dir /path/to/save \
    --num_samples 3000 \
    --batch_size 32
```

---

## 模型架构对比

### StyleGAN2 vs 原始 GAN
| 特性 | 原始 GAN | StyleGAN2 |
|------|---------|-----------|
| 映射网络 | ✗ | ✓ (8层 MLP) |
| 自适应归一化 | ✗ | ✓ (AdaIN) |
| 渐进式训练 | ✗ | ✓ |
| 噪声注入 | ✗ | ✓ |
| 路径长度正则化 | ✗ | ✓ |
| 权重均衡 | ✗ | ✓ |
| 生成质量 | 中 | **高** |

**核心优势**:
1. **更稳定的训练**: Wasserstein loss + R1 正则化
2. **更好的解耦**: Z → W 映射使得潜空间更线性
3. **更高的质量**: 多种正则化技术

### NVAE vs 原始 VAE
| 特性 | 原始 VAE | NVAE |
|------|---------|------|
| 潜变量层级 | 单层 | **多层** (3+) |
| 网络深度 | 浅 | **深** (ResNet-like) |
| 残差连接 | ✗ | ✓ |
| SE 模块 | ✗ | ✓ |
| Batch Norm | ✗ | ✓ |
| 训练稳定性 | 中 | **高** |
| 重建质量 | 中 | **高** |

**核心优势**:
1. **层次化潜空间**: 捕捉不同尺度的特征
2. **更深的网络**: 残差连接使得可以训练更深的模型
3. **更好的先验**: 多尺度 KL 散度

---

## 性能优化建议

### StyleGAN2
1. **Batch Size**: 
   - 单 GPU: 4-8
   - 多 GPU: 16-32 (总)
   
2. **学习率**:
   - Generator: 2e-3
   - Discriminator: 2e-3
   - 使用 Adam (β1=0.0, β2=0.99)

3. **正则化**:
   - R1 gradient penalty: λ=10.0
   - 每 4 步计算一次梯度惩罚以节省计算

4. **训练时间**:
   - 256×160: ~300K steps
   - 预计时间: 2-3 天 (4×V100)

### NVAE
1. **Batch Size**:
   - 单 GPU: 4-8
   - 多 GPU: 16-32 (总)

2. **学习率**:
   - 初始: 1e-3
   - Weight decay: 3e-4
   - 使用 AdamW

3. **KL Annealing**:
   - 建议从小权重开始: 1e-6
   - 逐渐增加到 1e-4

4. **梯度裁剪**:
   - 非常重要！阈值: 1.0
   - 防止梯度爆炸

5. **训练时间**:
   - 256×160: ~300K steps
   - 预计时间: 2-3 天 (4×V100)

---

## 常见问题

### Q1: StyleGAN2 训练不稳定
**A**: 
- 检查 R1 正则化是否启用
- 尝试降低学习率
- 增加 `--gp_lambda` (例如 15.0)

### Q2: NVAE 重建模糊
**A**:
- 降低 `--kl_weight` (例如 1e-5)
- 增加 `--num_scales` (例如 4)
- 增加 `--base_channels` (例如 96)

### Q3: 显存不足
**A**:
- 减小 `--batch_size`
- StyleGAN2: 减小 `--w_dim`
- NVAE: 减小 `--base_channels` 或 `--num_scales`

### Q4: 训练速度慢
**A**:
- 使用混合精度训练 (需要修改代码添加 `torch.cuda.amp`)
- 减少 checkpoint 保存频率
- StyleGAN2: 增加 `--gp_interval`

---

## 预期性能

基于典型的移动数据生成任务 (256×160 灰度图):

| 模型 | FID ↓ | IS ↑ | 训练时间 | 采样速度 |
|------|-------|------|---------|---------|
| 原始 GAN | ~45 | ~3.5 | 1 天 | 快 |
| StyleGAN2 | **~25** | **~5.0** | 3 天 | 中 |
| 原始 VAE | ~55 | ~3.0 | 1 天 | 快 |
| NVAE | **~30** | **~4.5** | 3 天 | 中 |

*注: 具体数值取决于数据集质量和超参数*

---

## 引用

如果使用这些模型，请引用原始论文：

**StyleGAN2**:
```bibtex
@inproceedings{karras2020analyzing,
  title={Analyzing and improving the image quality of stylegan},
  author={Karras, Tero and Laine, Samuli and Aittala, Miika and Hellsten, Janne and Lehtinen, Jaakko and Aila, Timo},
  booktitle={CVPR},
  year={2020}
}
```

**NVAE**:
```bibtex
@inproceedings{vahdat2020nvae,
  title={NVAE: A deep hierarchical variational autoencoder},
  author={Vahdat, Arash and Kautz, Jan},
  booktitle={NeurIPS},
  year={2020}
}
```

---

## 许可证

代码遵循 MIT 许可证。

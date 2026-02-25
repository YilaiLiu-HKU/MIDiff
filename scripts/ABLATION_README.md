# TripletAttention消融实验说明

## 概述

本实验旨在将TripletAttention机制复刻到Transformer-based的DiT模型中，通过消融实验验证TripletAttention在Transformer架构中的效果。

## 文件说明

1. **`triplet_attention_transformer.py`**: TripletAttention的Transformer适配实现
   - `TripletAttention2D`: 原始2D版本的TripletAttention
   - `TripletAttentionTransformer`: V1版本，将tokens重新组织成2D特征图
   - `TripletAttentionTransformerV2`: V2版本，直接在token空间应用注意力
   - `HybridAttention`: 混合注意力，结合self-attention和TripletAttention

2. **`dit_with_triplet.py`**: 支持TripletAttention的DiT模型
   - 修改了`Transformer`类以支持不同的attention类型
   - 支持4种attention模式：`origin`, `triplet_replace`, `triplet_add`, `hybrid`

3. **`ablation_experiment.py`**: 消融实验主脚本
   - 自动配置不同的实验设置
   - 保存实验配置和结果

4. **`run_ablation_experiments.sh`**: 批量运行脚本
   - 自动运行所有消融实验配置

## 消融实验配置

### 1. Baseline (baseline)
- **描述**: 原始DiT模型，使用标准self-attention
- **配置**: `attention_type='origin'`

### 2. TripletAttention替换 (triplet_replace)
- **描述**: 用TripletAttention完全替换self-attention
- **配置**: `attention_type='triplet_replace'`
- **实现**: 在Transformer层中使用TripletAttention替代标准Attention

### 3. TripletAttention添加 (triplet_add)
- **描述**: 在self-attention后额外添加TripletAttention
- **配置**: `attention_type='triplet_add'`
- **实现**: 先应用self-attention，再应用TripletAttention

### 4. 混合注意力 (hybrid)
- **描述**: 混合使用self-attention和TripletAttention
- **配置**: `attention_type='hybrid'`
- **实现**: 使用HybridAttention，同时应用两种注意力机制并融合

## 使用方法

### 单个实验

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/ablation_experiment.py \
    --data_dir /home/yilai/projects/poster/NetDiffus/tiff_log \
    --image_size 256 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --learn_sigma True \
    --class_cond False \
    --lr 5e-4 \
    --batch_size 32 \
    --save_dir ablation_experiments \
    --special_weight 1.0 \
    --cos_weight 0.01 \
    --save_interval 1000 \
    --ablation_config baseline \
    --triplet_version v1 \
    --triplet_no_spatial True
```

### 批量运行所有实验

```bash
chmod +x scripts/run_ablation_experiments.sh
./scripts/run_ablation_experiments.sh
```

## 参数说明

### 消融实验特定参数

- `--ablation_config`: 消融实验配置
  - `baseline`: 原始DiT
  - `triplet_replace`: TripletAttention替换
  - `triplet_add`: TripletAttention添加
  - `hybrid`: 混合注意力

- `--triplet_version`: TripletAttention实现版本
  - `v1`: 将tokens重新组织成2D特征图后应用TripletAttention
  - `v2`: 直接在token空间应用TripletAttention机制

- `--triplet_no_spatial`: 是否在TripletAttention中使用空间注意力
  - `True`: 只使用H轴和W轴注意力（默认）
  - `False`: 同时使用H轴、W轴和空间注意力

## TripletAttention机制说明

### 原始TripletAttention（2D版本）

TripletAttention通过三个方向的空间注意力来增强特征表示：

1. **H轴注意力**: 沿高度方向应用通道注意力
   ```python
   x_perm1 = x.permute(0, 2, 1, 3)  # (B, H, C, W)
   x_out1 = ChannelGateH(x_perm1)
   ```

2. **W轴注意力**: 沿宽度方向应用通道注意力
   ```python
   x_perm2 = x.permute(0, 3, 2, 1)  # (B, W, H, C)
   x_out2 = ChannelGateW(x_perm2)
   ```

3. **融合**: 平均融合并添加残差连接
   ```python
   x_out = (1/2) * (x_out1 + x_out2) + x
   ```

### Transformer适配版本

在Transformer中，tokens的形状是`(B, N, D)`，其中：
- `B`: batch size
- `N`: token数量（通常是`H * W`）
- `D`: token特征维度

**V1版本**将tokens重新组织成2D特征图`(B, D, H, W)`，然后应用原始TripletAttention。

**V2版本**直接在token空间应用TripletAttention机制，对H和W方向分别应用注意力。

## 实验记录

实验结果会保存在`ablation_experiments/`目录下，每个实验有独立的子目录：

```
ablation_experiments/
├── baseline_20240101_120000/
│   ├── baseline_config.json
│   ├── training_config.txt
│   └── ...
├── triplet_replace_v1_20240101_130000/
│   └── ...
└── ...
```

## 注意事项

1. **Token空间维度**: TripletAttention需要知道token的空间维度（H和W），这需要根据输入图像尺寸和patch size计算。

2. **模型兼容性**: 需要确保`script_util.py`中的`create_model`函数能够正确创建支持TripletAttention的DiT模型。

3. **内存使用**: TripletAttention可能会增加内存使用，特别是V1版本需要重新组织tokens。

4. **训练稳定性**: 不同配置的训练稳定性可能不同，建议监控训练损失。

## 下一步工作

1. 修改`script_util.py`以支持使用`dit_with_triplet.py`中的DiT
2. 实现动态token空间维度计算
3. 添加实验结果的自动对比和分析
4. 优化TripletAttention的实现以提高效率

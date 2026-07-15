# 模型架构分析

## 训练命令对应的模型配置

```bash
CUDA_VISIBLE_DEVICES=7 python train_midiff.py \
  --data_dir ./cgasf \
  --image_size 256 \
  --num_channels 128 \
  --num_res_blocks 3 \
  --diffusion_steps 1000 \
  --noise_schedule cosine \
  --learn_sigma True \
  --class_cond False \
  --rescale_learned_sigmas False \
  --rescale_timesteps False \
  --lr 5e-4 \
  --batch_size 32 \
  --save_dir ckpt/cgasf \
  --special_weight 1.0 \
  --attention_type triple \
  --save_interval 1000
```

## 模型架构详解

### 1. 整体架构：UNet + ResNet Backbone

- **Backbone类型**: ResNet (默认 `backbone_type='resnet'`)
- **模型类型**: UNet (默认 `backbone='unet'`)
- **输入**: 单通道图像 (1, 256, 256)
- **输出**: 2通道 (均值和方差，因为 `learn_sigma=True`)

### 2. 通道配置

- **基础通道数**: `model_channels = 128`
- **通道倍数**: `channel_mult = (1, 1, 2, 2, 4, 4)` (针对256x256图像)
- **各层通道数**:
  - Level 0: 128 × 1 = 128
  - Level 1: 128 × 1 = 128
  - Level 2: 128 × 2 = 256
  - Level 3: 128 × 2 = 256
  - Level 4: 128 × 4 = 512
  - Level 5: 128 × 4 = 512

### 3. 残差块配置

- **每层残差块数**: `num_res_blocks = 3`
- **ResBlock结构**:
  ```
  ResBlock(
    - 输入层: GroupNorm + SiLU + Conv2d(3x3)
    - 时间嵌入: SiLU + Linear (嵌入到通道维度)
    - 输出层: GroupNorm + SiLU + Dropout + Conv2d(3x3, zero初始化)
    - 跳跃连接: Conv2d(1x1) 或 Identity
  )
  ```

### 4. 注意力机制：TripletAttention

- **注意力类型**: `attention_type='triple'` (使用TripletAttention)
- **注意力分辨率**: `attention_resolutions = [16, 32]` (默认 "16,8")
  - 对于256x256图像: [256//16, 256//8] = [16, 32]
- **TripletAttention工作原理**:
  ```
  1. ChannelGateH: 沿H轴(高度)的通道注意力
     - 输入: (B, C, H, W) -> permute(0,2,1,3) -> (B, H, C, W)
     - 使用SpatialGate进行空间注意力
     - 输出: (B, H, C, W) -> permute(0,2,1,3) -> (B, C, H, W)
  
  2. ChannelGateW: 沿W轴(宽度)的通道注意力
     - 输入: (B, C, H, W) -> permute(0,3,2,1) -> (B, W, H, C)
     - 使用SpatialGate进行空间注意力
     - 输出: (B, W, H, C) -> permute(0,3,2,1) -> (B, C, H, W)
  
  3. 融合: out = (1/2) * (out_H + out_W) + x (残差连接)
  ```

### 5. UNet结构层次

#### 编码器 (Input Blocks)
```
输入: (B, 1, 256, 256)
  ↓
Conv2d(1 -> 128, 3x3)  [256x256, 128ch]
  ↓
Level 0: 3×ResBlock(128->128) + TripletAttention(在16x16时)  [256x256, 128ch]
  ↓ Downsample
Level 1: 3×ResBlock(128->128) + TripletAttention(在16x16时)  [128x128, 128ch]
  ↓ Downsample
Level 2: 3×ResBlock(128->256) + TripletAttention(在32x32时)  [64x64, 256ch]
  ↓ Downsample
Level 3: 3×ResBlock(256->256) + TripletAttention(在32x32时)  [32x32, 256ch]
  ↓ Downsample
Level 4: 3×ResBlock(256->512)  [16x16, 512ch]
  ↓ Downsample
Level 5: 3×ResBlock(512->512)  [8x8, 512ch]
```

#### 中间层 (Middle Block)
```
ResBlock(512->512)
  ↓
TripletAttention(512ch)
  ↓
ResBlock(512->512)
```

#### 解码器 (Output Blocks)
```
Level 5: Concat[512, 512] -> ResBlock(1024->512)  [8x8, 512ch]
  ↓ Upsample
Level 4: Concat[512, 512] -> ResBlock(1024->512)  [16x16, 512ch]
  ↓ Upsample
Level 3: Concat[512, 256] -> ResBlock(768->256) + TripletAttention  [32x32, 256ch]
  ↓ Upsample
Level 2: Concat[256, 256] -> ResBlock(512->256) + TripletAttention  [64x64, 256ch]
  ↓ Upsample
Level 1: Concat[256, 128] -> ResBlock(384->128) + TripletAttention  [128x128, 128ch]
  ↓ Upsample
Level 0: Concat[128, 128] -> ResBlock(256->128) + TripletAttention  [256x256, 128ch]
  ↓
输出层: GroupNorm + SiLU + Conv2d(128->2, 3x3)
输出: (B, 2, 256, 256)  [均值和方差]
```

### 6. 时间步嵌入

- **嵌入维度**: `time_embed_dim = model_channels * 4 = 512`
- **结构**:
  ```
  Linear(128 -> 512)
    ↓
  SiLU
    ↓
  Linear(512 -> 512)
  ```
- **时间步嵌入方式**: 使用正弦位置编码

### 7. 扩散过程配置

- **扩散步数**: 1000步
- **噪声调度**: Cosine schedule
- **模型输出**: 均值和方差 (learn_sigma=True)
- **方差类型**: LEARNED_RANGE (学习方差范围)

### 8. 关键特性

1. **TripletAttention的优势**:
   - 同时考虑H轴和W轴的空间关系
   - 轻量级，计算效率高
   - 通过残差连接保持梯度流

2. **ResNet Backbone**:
   - 使用GroupNorm进行归一化
   - SiLU激活函数
   - 时间步条件注入 (通过scale-shift或加法)

3. **多尺度特征融合**:
   - UNet的跳跃连接保留细节信息
   - 在不同分辨率层使用注意力机制

### 9. 损失函数

- **主要损失**: MSE损失 (预测噪声)
- **特殊权重**: `special_weight=1.0` (对有语义区域加权)
- **余弦损失**: `cos_weight=0.01` (余弦相似度损失)

### 10. 模型参数量估算

- **基础通道**: 128
- **最大通道**: 512
- **总层数**: 约20+层 (编码器+解码器+中间层)
- **参数量**: 约数百万到数千万参数 (取决于具体实现)

## 模型特点总结

1. **轻量级注意力**: TripletAttention比标准自注意力更高效
2. **多尺度处理**: 在不同分辨率层应用注意力
3. **条件生成**: 通过时间步嵌入控制生成过程
4. **方差学习**: 同时学习均值和方差，提高生成质量
5. **残差连接**: 保持梯度流，便于训练深层网络

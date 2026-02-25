"""
StyleGAN2 核心模块实现
支持非正方形图像 (256x160)
参考: https://github.com/NVlabs/stylegan2-ada-pytorch
"""

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class EqualizedLinear(nn.Module):
    """权重均衡的全连接层"""
    def __init__(self, in_features, out_features, bias=True, bias_init=0, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(th.randn(out_features, in_features) / lr_mul)
        if bias:
            self.bias = nn.Parameter(th.full([out_features], float(bias_init)))
        else:
            self.bias = None
        self.weight_gain = lr_mul / math.sqrt(in_features)
        self.bias_gain = lr_mul

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain
        return F.linear(x, w, b)


class EqualizedConv2d(nn.Module):
    """权重均衡的2D卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            th.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(th.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        self.weight_gain = 1.0 / math.sqrt(fan_in)

    def forward(self, x):
        w = self.weight * self.weight_gain
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)


class ModulatedConv2d(nn.Module):
    """StyleGAN2 的调制卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, 
                 demodulate=True, upsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        
        # 仿射变换：从 style (w) 生成调制参数
        self.affine = EqualizedLinear(style_dim, in_channels, bias_init=1)
        
        # 卷积权重
        self.weight = nn.Parameter(
            th.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        fan_in = in_channels * kernel_size * kernel_size
        self.weight_gain = 1.0 / math.sqrt(fan_in)

    def forward(self, x, style):
        batch, in_c, h, w = x.shape
        
        # 1. 计算调制参数
        s = self.affine(style)  # [B, in_c]
        
        # 2. 调制权重
        weight = self.weight * self.weight_gain  # [out_c, in_c, k, k]
        weight = weight.unsqueeze(0)  # [1, out_c, in_c, k, k]
        weight = weight * s.view(batch, 1, in_c, 1, 1)  # [B, out_c, in_c, k, k]
        
        # 3. 去调制（Demodulation）
        if self.demodulate:
            d = th.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * d
        
        # 4. 重塑权重用于分组卷积
        weight = weight.view(batch * self.out_channels, in_c, self.kernel_size, self.kernel_size)
        
        # 5. 上采样（如果需要）
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            h, w = h * 2, w * 2
        
        # 6. 分组卷积
        x = x.view(1, batch * in_c, h, w)
        x = F.conv2d(x, weight, padding=self.kernel_size//2, groups=batch)
        x = x.view(batch, self.out_channels, x.shape[2], x.shape[3])
        
        return x


class NoiseInjection(nn.Module):
    """噪声注入层"""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(th.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = th.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return x + self.weight.view(1, -1, 1, 1) * noise


class StyleBlock(nn.Module):
    """StyleGAN2 的合成块"""
    def __init__(self, in_channels, out_channels, style_dim, upsample=False):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, style_dim, 
                                      upsample=upsample)
        self.noise1 = NoiseInjection(out_channels)
        self.act1 = nn.LeakyReLU(0.2)
        
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, style_dim)
        self.noise2 = NoiseInjection(out_channels)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise1=None, noise2=None):
        x = self.conv1(x, style)
        x = self.noise1(x, noise1)
        x = self.act1(x)
        
        x = self.conv2(x, style)
        x = self.noise2(x, noise2)
        x = self.act2(x)
        return x


class MappingNetwork(nn.Module):
    """映射网络：Z -> W"""
    def __init__(self, z_dim=256, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(EqualizedLinear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        # 归一化输入
        z = F.normalize(z, dim=1)
        return self.net(z)


class SynthesisNetwork(nn.Module):
    """合成网络：W -> Image (支持非正方形)"""
    def __init__(self, w_dim=512, img_channels=1, img_size_h=256, img_size_w=160, 
                 channel_base=32768, channel_max=512):
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.w_dim = w_dim
        
        # 计算需要的层数
        self.num_layers_h = int(np.log2(img_size_h))
        self.num_layers_w = int(np.log2(img_size_w))
        self.num_layers = max(self.num_layers_h, self.num_layers_w)
        
        # 起始分辨率
        self.start_res_h = img_size_h // (2 ** (self.num_layers - 2))
        self.start_res_w = img_size_w // (2 ** (self.num_layers - 2))
        
        def nf(stage):
            """计算通道数"""
            return min(int(channel_base / (2.0 ** stage)), channel_max)
        
        # 常量输入
        self.const = nn.Parameter(
            th.randn(1, nf(0), self.start_res_h, self.start_res_w)
        )
        
        # 构建合成块
        self.blocks = nn.ModuleList()
        in_ch = nf(0)
        
        for i in range(self.num_layers - 2):
            out_ch = nf(i + 1)
            upsample = i > 0
            self.blocks.append(StyleBlock(in_ch, out_ch, w_dim, upsample=upsample))
            in_ch = out_ch
        
        # ToRGB 层
        self.to_rgb = EqualizedConv2d(in_ch, img_channels, 1)

    def forward(self, w):
        batch = w.shape[0]
        x = self.const.repeat(batch, 1, 1, 1)
        
        for block in self.blocks:
            x = block(x, w)
        
        # 确保输出尺寸正确
        if x.shape[2] != self.img_size_h or x.shape[3] != self.img_size_w:
            x = F.interpolate(x, size=(self.img_size_h, self.img_size_w), 
                            mode='bilinear', align_corners=False)
        
        x = self.to_rgb(x)
        return x


class StyleGAN2Generator(nn.Module):
    """完整的 StyleGAN2 生成器"""
    def __init__(self, z_dim=256, w_dim=512, img_channels=1, 
                 img_size_h=256, img_size_w=160):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        self.mapping = MappingNetwork(z_dim, w_dim, num_layers=8)
        self.synthesis = SynthesisNetwork(w_dim, img_channels, img_size_h, img_size_w)
        
    def forward(self, z):
        w = self.mapping(z)
        img = self.synthesis(w)
        return th.tanh(img)


class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 判别器（支持非正方形）"""
    def __init__(self, img_channels=1, img_size_h=256, img_size_w=160, 
                 channel_base=16384, channel_max=512):
        super().__init__()
        
        def nf(stage):
            return min(int(channel_base / (2.0 ** stage)), channel_max)
        
        num_layers = int(np.log2(min(img_size_h, img_size_w)))
        
        # FromRGB
        self.from_rgb = EqualizedConv2d(img_channels, nf(num_layers-1), 1)
        
        # 下采样块
        blocks = []
        in_ch = nf(num_layers-1)
        for i in range(num_layers-1, 0, -1):
            out_ch = nf(i-1)
            blocks.append(nn.Sequential(
                EqualizedConv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        
        # 最终层（使用全局池化避免尺寸问题）
        self.final = nn.Sequential(
            EqualizedConv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Flatten(),
            EqualizedLinear(in_ch, 1)
        )
        
    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)
        x = self.blocks(x)
        x = self.final(x)
        return x.squeeze(-1)  # 返回 [B]

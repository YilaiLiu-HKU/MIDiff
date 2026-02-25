"""
NVAE (Nouveau VAE) 核心模块实现
支持非正方形图像 (256x160)
参考: https://github.com/NVlabs/NVAE
"""

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Swish(nn.Module):
    """Swish 激活函数"""
    def forward(self, x):
        return x * th.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-Excitation 模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualCell(nn.Module):
    """NVAE 的残差单元"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
        self.se = SE(out_channels)
        self.act = Swish()
        
        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += residual
        out = self.act(out)
        return out


class EncoderResidualBlock(nn.Module):
    """编码器残差块"""
    def __init__(self, in_channels, out_channels, num_cells=2, downsample=False):
        super().__init__()
        cells = []
        for i in range(num_cells):
            stride = 2 if downsample and i == 0 else 1
            in_ch = in_channels if i == 0 else out_channels
            cells.append(ResidualCell(in_ch, out_channels, stride))
        self.cells = nn.Sequential(*cells)

    def forward(self, x):
        return self.cells(x)


class DecoderResidualBlock(nn.Module):
    """解码器残差块"""
    def __init__(self, in_channels, out_channels, num_cells=2, upsample=False):
        super().__init__()
        self.upsample = upsample
        
        cells = []
        for i in range(num_cells):
            in_ch = in_channels if i == 0 else out_channels
            cells.append(ResidualCell(in_ch, out_channels, stride=1))
        self.cells = nn.Sequential(*cells)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.cells(x)


class NVAEEncoder(nn.Module):
    """NVAE 编码器（多尺度潜变量）"""
    def __init__(self, in_channels=1, base_channels=64, num_scales=3, 
                 num_cells_per_scale=2, latent_dim=256):
        super().__init__()
        self.num_scales = num_scales
        self.latent_dim = latent_dim
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels, eps=1e-5, momentum=0.05),
            Swish()
        )
        
        # 多尺度编码块
        self.encoder_blocks = nn.ModuleList()
        self.pre_z_convs = nn.ModuleList()  # 用于潜变量预测
        
        ch = base_channels
        for scale in range(num_scales):
            downsample = scale > 0
            next_ch = ch * 2 if downsample else ch
            
            block = EncoderResidualBlock(ch, next_ch, num_cells_per_scale, downsample)
            self.encoder_blocks.append(block)
            
            # 每个尺度预测潜变量的 mu 和 logvar
            self.pre_z_convs.append(nn.Conv2d(next_ch, latent_dim * 2, 1))
            
            ch = next_ch
        
        # 全局池化用于顶层潜变量
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        返回: List[(mu, logvar, spatial_size)] for each scale
        """
        h = self.stem(x)
        
        latent_stats = []
        for i, (block, pre_z) in enumerate(zip(self.encoder_blocks, self.pre_z_convs)):
            h = block(h)
            stats = pre_z(h)
            mu, logvar = th.chunk(stats, 2, dim=1)
            
            # 顶层使用全局池化
            if i == self.num_scales - 1:
                mu = self.global_pool(mu).squeeze(-1).squeeze(-1)
                logvar = self.global_pool(logvar).squeeze(-1).squeeze(-1)
            
            latent_stats.append((mu, logvar, h.shape[2:]))
        
        return latent_stats


class NVAEDecoder(nn.Module):
    """NVAE 解码器（多尺度潜变量）"""
    def __init__(self, out_channels=1, base_channels=64, num_scales=3, 
                 num_cells_per_scale=2, latent_dim=256, img_size_h=256, img_size_w=160):
        super().__init__()
        self.num_scales = num_scales
        self.latent_dim = latent_dim
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.base_channels = base_channels
        
        # 计算最小特征图尺寸
        self.min_h = img_size_h // (2 ** (num_scales - 1))
        self.min_w = img_size_w // (2 ** (num_scales - 1))
        
        # 顶层通道数
        self.top_ch = base_channels * (2 ** (num_scales - 1))
        
        # 将顶层潜变量投影到特征图
        self.z_proj = nn.Linear(latent_dim, self.top_ch * self.min_h * self.min_w)
        
        # 多尺度解码块
        self.decoder_blocks = nn.ModuleList()
        self.post_z_convs = nn.ModuleList()  # 合并潜变量
        
        ch = self.top_ch
        for scale in range(num_scales - 1, -1, -1):
            upsample = scale > 0
            next_ch = ch // 2 if upsample else ch
            
            # 用于合并该尺度的潜变量
            if scale < num_scales - 1:
                self.post_z_convs.append(nn.Conv2d(latent_dim + ch, ch, 1))
            
            block = DecoderResidualBlock(ch, next_ch, num_cells_per_scale, upsample)
            self.decoder_blocks.append(block)
            
            ch = next_ch
        
        # 输出头
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch, eps=1e-5, momentum=0.05),
            Swish(),
            nn.Conv2d(ch, out_channels, 3, 1, 1)
        )

    def forward(self, latent_samples):
        """
        latent_samples: List[(z, spatial_size)] for each scale (从顶到底)
        """
        # 顶层潜变量
        z_top = latent_samples[-1][0]  # (B, latent_dim)
        batch_size = z_top.shape[0]
        
        # 投影并重塑
        h = self.z_proj(z_top)  # (B, top_ch * min_h * min_w)
        h = h.view(batch_size, self.top_ch, self.min_h, self.min_w)
        
        # 逐层解码
        conv_idx = 0
        for i, block in enumerate(self.decoder_blocks):
            scale_idx = self.num_scales - 1 - i
            
            # 合并该尺度的潜变量（除了顶层）
            if scale_idx < self.num_scales - 1:
                z_cur, size_cur = latent_samples[scale_idx]
                # 调整 z 到当前空间尺寸
                if len(z_cur.shape) == 2:  # 全局潜变量
                    z_cur = z_cur.unsqueeze(-1).unsqueeze(-1)
                    z_cur = z_cur.expand(-1, -1, h.shape[2], h.shape[3])
                elif z_cur.shape[2:] != h.shape[2:]:
                    z_cur = F.interpolate(z_cur, size=h.shape[2:], mode='bilinear', 
                                        align_corners=False)
                
                h = th.cat([h, z_cur], dim=1)
                h = self.post_z_convs[conv_idx](h)
                conv_idx += 1
            
            h = block(h)
        
        # 确保输出尺寸正确
        if h.shape[2] != self.img_size_h or h.shape[3] != self.img_size_w:
            h = F.interpolate(h, size=(self.img_size_h, self.img_size_w), 
                            mode='bilinear', align_corners=False)
        
        out = self.head(h)
        return out


class NVAE(nn.Module):
    """完整的 NVAE 模型"""
    def __init__(self, in_channels=1, base_channels=64, num_scales=3, 
                 num_cells_per_scale=2, latent_dim=256, img_size_h=256, img_size_w=160):
        super().__init__()
        self.encoder = NVAEEncoder(in_channels, base_channels, num_scales, 
                                   num_cells_per_scale, latent_dim)
        self.decoder = NVAEDecoder(in_channels, base_channels, num_scales, 
                                   num_cells_per_scale, latent_dim, img_size_h, img_size_w)
        self.num_scales = num_scales

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """训练前向传播"""
        # 编码
        latent_stats = self.encoder(x)
        
        # 采样
        latent_samples = []
        kl_losses = []
        for mu, logvar, size in latent_stats:
            z = self.reparameterize(mu, logvar)
            latent_samples.append((z, size))
            
            # 计算 KL 散度
            kl = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_losses.append(kl.mean())
        
        # 解码
        recons = self.decoder(latent_samples)
        recons = th.tanh(recons)
        
        return recons, kl_losses

    def sample(self, batch_size, device):
        """从先验采样"""
        latent_samples = []
        
        for scale in range(self.num_scales):
            # 从标准正态分布采样
            if scale == self.num_scales - 1:
                # 顶层是全局潜变量
                z = th.randn(batch_size, self.encoder.latent_dim, device=device)
                size = None
            else:
                # 其他层是空间潜变量
                h = self.decoder.img_size_h // (2 ** (self.num_scales - 1 - scale))
                w = self.decoder.img_size_w // (2 ** (self.num_scales - 1 - scale))
                z = th.randn(batch_size, self.encoder.latent_dim, h, w, device=device)
                size = (h, w)
            
            latent_samples.append((z, size))
        
        # 解码
        with th.no_grad():
            samples = self.decoder(latent_samples)
            samples = th.tanh(samples)
        
        return samples

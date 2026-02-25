"""
TripletAttention的Transformer适配版本
将TripletAttention机制复刻到transformer based模型中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BasicConv(nn.Module):
    """基础卷积模块，用于SpatialGate"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    """通道池化：最大池化和平均池化的拼接"""
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    """空间门控机制"""
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class TripletAttention2D(nn.Module):
    """
    2D版本的TripletAttention，用于处理特征图 (B, C, H, W)
    这是原始TripletAttention的实现
    """
    def __init__(self, gate_channels, no_spatial=True):
        super(TripletAttention2D, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        # x shape: (B, C, H, W)
        # H轴注意力
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()  # (B, H, C, W)
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()  # (B, C, H, W)
        
        # W轴注意力
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()  # (B, W, H, C)
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()  # (B, C, H, W)
        
        # 融合
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        
        return x_out + x  # 残差连接


class TripletAttentionTransformer(nn.Module):
    """
    TripletAttention的Transformer适配版本
    将tokens重新组织成2D特征图，应用TripletAttention，然后恢复为tokens
    """
    def __init__(self, dim, num_tokens_h, num_tokens_w, no_spatial=True):
        """
        Args:
            dim: token的特征维度
            num_tokens_h: token在高度方向的数量（需要知道原始图像的高度）
            num_tokens_w: token在宽度方向的数量（需要知道原始图像的宽度）
            no_spatial: 是否使用空间注意力
        """
        super(TripletAttentionTransformer, self).__init__()
        self.dim = dim
        self.num_tokens_h = num_tokens_h
        self.num_tokens_w = num_tokens_w
        
        # 将TripletAttention应用到特征维度上
        # 注意：这里我们将每个token的特征维度视为"通道"
        self.triplet_attn = TripletAttention2D(gate_channels=dim, no_spatial=no_spatial)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, D) tokens，其中N = num_tokens_h * num_tokens_w
        Returns:
            out: (B, N, D) tokens
        """
        B, N, D = x.shape
        assert N == self.num_tokens_h * self.num_tokens_w, \
            f"Token数量 {N} 与预期 {self.num_tokens_h * self.num_tokens_w} 不匹配"
        assert D == self.dim, f"特征维度 {D} 与预期 {self.dim} 不匹配"
        
        # 将tokens重新组织成2D特征图
        # (B, N, D) -> (B, D, H, W)
        # 这里我们将特征维度D视为"通道"，将token空间视为H×W
        x_2d = x.transpose(1, 2).contiguous()  # (B, D, N)
        x_2d = x_2d.view(B, D, self.num_tokens_h, self.num_tokens_w)  # (B, D, H, W)
        
        # 应用TripletAttention
        x_2d_out = self.triplet_attn(x_2d)  # (B, D, H, W)
        
        # 恢复为tokens
        x_out = x_2d_out.view(B, D, N).transpose(1, 2).contiguous()  # (B, N, D)
        
        return x_out


class TripletAttentionTransformerV2(nn.Module):
    """
    TripletAttention的Transformer适配版本 V2
    另一种实现方式：直接在token空间上应用TripletAttention机制
    将token序列视为2D网格，对H和W方向分别应用注意力
    """
    def __init__(self, dim, num_tokens_h, num_tokens_w, no_spatial=True):
        super(TripletAttentionTransformerV2, self).__init__()
        self.dim = dim
        self.num_tokens_h = num_tokens_h
        self.num_tokens_w = num_tokens_w
        self.no_spatial = no_spatial
        
        # H轴注意力：对每一列（W个tokens）应用注意力
        self.h_attention = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # W轴注意力：对每一行（H个tokens）应用注意力
        self.w_attention = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        if not no_spatial:
            # 空间注意力：对所有tokens应用注意力
            self.spatial_attention = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D) tokens
        Returns:
            out: (B, N, D) tokens
        """
        B, N, D = x.shape
        assert N == self.num_tokens_h * self.num_tokens_w
        
        # 重新组织为2D: (B, N, D) -> (B, H, W, D)
        x_2d = x.view(B, self.num_tokens_h, self.num_tokens_w, D)
        
        # H轴注意力：对每一列应用
        # (B, H, W, D) -> (B*W, H, D) -> 应用注意力 -> (B*W, H, D) -> (B, H, W, D)
        x_h = x_2d.permute(0, 2, 1, 3).contiguous()  # (B, W, H, D)
        x_h = x_h.view(B * self.num_tokens_w, self.num_tokens_h, D)
        h_weights = self.h_attention(x_h)  # (B*W, H, D)
        x_h = x_h * h_weights
        x_h = x_h.view(B, self.num_tokens_w, self.num_tokens_h, D).permute(0, 2, 1, 3).contiguous()  # (B, H, W, D)
        
        # W轴注意力：对每一行应用
        # (B, H, W, D) -> (B*H, W, D) -> 应用注意力 -> (B*H, W, D) -> (B, H, W, D)
        x_w = x_2d.view(B * self.num_tokens_h, self.num_tokens_w, D)
        w_weights = self.w_attention(x_w)  # (B*H, W, D)
        x_w = x_w * w_weights
        x_w = x_w.view(B, self.num_tokens_h, self.num_tokens_w, D)
        
        # 融合
        if not self.no_spatial:
            spatial_weights = self.spatial_attention(x_2d)  # (B, H, W, D)
            x_out = (1 / 3) * (x_h + x_w + x_2d * spatial_weights)
        else:
            x_out = (1 / 2) * (x_h + x_w)
        
        # 恢复为tokens并添加残差
        x_out = x_out.view(B, N, D)
        return x_out + x


class HybridAttention(nn.Module):
    """
    混合注意力：结合标准self-attention和TripletAttention
    """
    def __init__(self, dim, heads, dim_head, num_tokens_h, num_tokens_w, 
                 dropout=0., use_triplet=True, triplet_version='v1', no_spatial=True):
        super(HybridAttention, self).__init__()
        self.use_triplet = use_triplet
        
        # 标准self-attention
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout_attn = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if not (heads == 1 and dim_head == dim) else nn.Identity()
        
        # TripletAttention
        if use_triplet:
            if triplet_version == 'v1':
                self.triplet_attn = TripletAttentionTransformer(
                    dim, num_tokens_h, num_tokens_w, no_spatial=no_spatial
                )
            else:
                self.triplet_attn = TripletAttentionTransformerV2(
                    dim, num_tokens_h, num_tokens_w, no_spatial=no_spatial
                )
        else:
            self.triplet_attn = None
    
    def forward(self, x):
        # 标准self-attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout_attn(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        sa_out = self.to_out(out)
        
        # 如果使用TripletAttention，则融合
        if self.use_triplet and self.triplet_attn is not None:
            triplet_out = self.triplet_attn(x)
            # 融合策略：可以相加、拼接后投影、或加权平均
            # 这里使用简单的加权平均
            out = 0.5 * sa_out + 0.5 * triplet_out
        else:
            out = sa_out
        
        return out

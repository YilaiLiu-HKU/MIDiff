"""
DiT模型，支持TripletAttention机制
"""

import torch
import torch.nn as nn
import math
from einops import rearrange
from triplet_attention_transformer import (
    TripletAttentionTransformer,
    TripletAttentionTransformerV2,
    HybridAttention
)


class HeatmapProcessor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, hidden_size, 1)
        )
        
    def forward(self, heatmap):
        B, _, H, W = heatmap.shape
        max_val = heatmap.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        normalized_heatmap = heatmap / (max_val + 1e-6)
        binary_mask = torch.bernoulli(normalized_heatmap)
        features = self.processor(binary_mask)
        return torch.mean(features, dim=(2, 3))


class AdaLNZero(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
    
    def forward(self, x, condition=None, **kwargs):
        if condition is None:
            return self.fn(self.norm(x), **kwargs)
        gamma = self.gamma(condition).unsqueeze(1)
        beta = self.beta(condition).unsqueeze(1)
        normalized = self.norm(x)
        modulated = (1 + gamma) * normalized + beta
        return self.fn(modulated, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """标准self-attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """支持TripletAttention的Transformer"""
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim, 
        dropout=0., 
        use_heatmap=False,
        attention_type='origin',  # 'origin', 'triplet_replace', 'triplet_add', 'hybrid'
        num_tokens_h=None,
        num_tokens_w=None,
        triplet_version='v1',
        triplet_no_spatial=True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        norm_class = AdaLNZero if use_heatmap else PreNorm
        
        # 计算token的空间维度（如果提供）
        self.num_tokens_h = num_tokens_h
        self.num_tokens_w = num_tokens_w
        self.attention_type = attention_type
        
        for _ in range(depth):
            # 根据attention_type选择注意力机制
            if attention_type == 'origin':
                # 原始self-attention
                attn_module = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            elif attention_type == 'triplet_replace':
                # 用TripletAttention替换self-attention
                if triplet_version == 'v1':
                    attn_module = TripletAttentionTransformer(
                        dim, num_tokens_h, num_tokens_w, no_spatial=triplet_no_spatial
                    )
                else:
                    attn_module = TripletAttentionTransformerV2(
                        dim, num_tokens_h, num_tokens_w, no_spatial=triplet_no_spatial
                    )
            elif attention_type == 'triplet_add':
                # 在self-attention后添加TripletAttention
                attn_module = nn.Sequential(
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    TripletAttentionTransformer(
                        dim, num_tokens_h, num_tokens_w, no_spatial=triplet_no_spatial
                    ) if triplet_version == 'v1' else TripletAttentionTransformerV2(
                        dim, num_tokens_h, num_tokens_w, no_spatial=triplet_no_spatial
                    )
                )
            elif attention_type == 'hybrid':
                # 混合注意力
                attn_module = HybridAttention(
                    dim, heads, dim_head, num_tokens_h, num_tokens_w,
                    dropout=dropout, use_triplet=True, 
                    triplet_version=triplet_version, 
                    no_spatial=triplet_no_spatial
                )
            else:
                raise ValueError(f"未知的attention_type: {attention_type}")
            
            self.layers.append(nn.ModuleList([
                norm_class(dim, attn_module),
                norm_class(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, condition=None):
        for attn, ff in self.layers:
            x = attn(x, condition=condition) + x
            x = ff(x, condition=condition) + x
        return x


class DiT(nn.Module):
    """支持TripletAttention的DiT模型"""
    def get_timestep_embedding(self, timesteps, embedding_dim, max_positions=10000):
        half_dim = embedding_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def __init__(
        self,
        *,
        input_size=192,
        patch_size=20,
        in_channels=1,
        out_channels=1,
        hidden_size=1024,
        depth=24,
        heads=16,
        mlp_ratio=4.,
        drop_path=0.,
        use_heatmap=False,
        attention_type='origin',  # 新增：支持TripletAttention
        triplet_version='v1',  # 新增：TripletAttention版本
        triplet_no_spatial=True,  # 新增：是否使用空间注意力
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # 计算token的空间维度
        # 假设输入图像是 (H, W)，每行分成patch_size个token
        # 所以 num_tokens_h = H, num_tokens_w = patch_size
        # 但实际需要根据输入图像尺寸计算
        # 这里先假设，实际使用时需要根据输入调整
        self.tokens_per_row = patch_size
        patch_dim = in_channels * (input_size // patch_size)
        
        # 计算token空间维度（用于TripletAttention）
        # 假设输入是 (H, W)，每行分成patch_size个token
        # 需要根据实际输入图像尺寸计算
        # 这里使用默认值，实际应该从forward中获取
        self.num_tokens_h = None  # 将在forward中设置
        self.num_tokens_w = patch_size
        
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        self.time_dim = hidden_size
        self.time_embedding = nn.Sequential(
            nn.Linear(self.time_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.use_heatmap = use_heatmap
        self.heatmap_processor = HeatmapProcessor(hidden_size) if use_heatmap else None
        
        # 创建Transformer，传入attention_type和相关参数
        self.transformer = Transformer(
            dim=hidden_size,
            depth=depth,
            heads=heads,
            dim_head=hidden_size // heads,
            mlp_dim=int(hidden_size * mlp_ratio),
            dropout=drop_path,
            use_heatmap=use_heatmap,
            attention_type=attention_type,
            num_tokens_h=None,  # 将在forward中动态设置
            num_tokens_w=self.num_tokens_w,
            triplet_version=triplet_version,
            triplet_no_spatial=triplet_no_spatial
        )

        self.to_pixels = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 8 * out_channels)
        )
        
        if input_size == 256:
            input_size = 160
        adaptive_hidden = input_size * out_channels * 2
        self.row_adaptive = nn.Sequential(
            nn.LayerNorm(input_size * out_channels),
            nn.Linear(input_size * out_channels, adaptive_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(adaptive_hidden, adaptive_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(adaptive_hidden, input_size * out_channels),
            nn.LayerNorm(input_size * out_channels)
        )
        
        self.row_attention = nn.Sequential(
            nn.Linear(input_size * out_channels, input_size * out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, time_emb, heatmap=None):
        b, c, h, w = x.shape
        
        # 计算token空间维度
        num_tokens_h = h  # 每行对应一个token序列
        num_tokens_w = self.tokens_per_row
        
        # 动态更新transformer中的num_tokens_h（如果支持）
        # 注意：这里需要修改Transformer以支持动态设置
        # 暂时使用固定值，或者通过其他方式传递
        
        patch_width = w // self.tokens_per_row
        x = rearrange(x, 'b c h (n w) -> b (h n) (w c)', n=self.tokens_per_row, w=patch_width)
        x = self.to_patch_embedding(x)

        t_emb = self.get_timestep_embedding(time_emb, self.time_dim)
        time_tokens = self.time_embedding(t_emb)
        x = x + time_tokens.unsqueeze(1)
        
        condition = None
        if heatmap is not None and self.heatmap_processor is not None:
            condition = self.heatmap_processor(heatmap)
        
        # 在调用transformer之前，需要更新num_tokens_h
        # 这里我们通过修改transformer的forward来传递
        # 或者创建一个包装器
        x = self.transformer(x, condition=condition)

        x = self.to_pixels(x)
        x = rearrange(x, 'b (h n) (w c) -> b c h (n w)', 
                     h=h, n=self.tokens_per_row, 
                     w=patch_width, c=self.out_channels)
        
        x = x.contiguous().reshape(b * c * h, -1)
        attention_weights = self.row_attention(x)
        x_adapted = self.row_adaptive(x)
        x_adapted = x_adapted * attention_weights
        x_adapted = x_adapted + x
        x = x_adapted.view(b, c * self.out_channels // self.in_channels, h, -1)
        
        return x, None

import torch
import torch.nn as nn
import math
from einops import rearrange

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
        
        # 执行max归一化
        max_val = heatmap.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        normalized_heatmap = heatmap / (max_val + 1e-6)  # 添加eps避免除0
        
        # 使用归一化后的值作为伯努利分布的概率进行采样
        binary_mask = torch.bernoulli(normalized_heatmap)
        

        
        # 处理heatmap
        features = self.processor(binary_mask)
        
        # 全局平均池化得到条件向量 (B, hidden_size)
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
            # 如果没有条件，就使用普通的LayerNorm
            return self.fn(self.norm(x), **kwargs)
        
        # 生成gamma和beta
        gamma = self.gamma(condition).unsqueeze(1)  # (B, 1, dim)
        beta = self.beta(condition).unsqueeze(1)    # (B, 1, dim)
        
        # AdaLN-Zero: (1 + gamma) * LN(x) + beta
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
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., use_heatmap=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        norm_class = AdaLNZero if use_heatmap else PreNorm

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                norm_class(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                norm_class(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, condition=None):
        for attn, ff in self.layers:
            x = attn(x, condition=condition) + x
            x = ff(x, condition=condition) + x
        return x

class DiT(nn.Module):
    def get_timestep_embedding(self, timesteps, embedding_dim, max_positions=10000):
        # 将时间步编码为高维向量，使用位置编码的方法
        half_dim = embedding_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # 如果维度是奇数，在最后添加一个零
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def __init__(
        self,
        *,
        input_size=192,
        patch_size=20,  # 在1D情况下，这代表每行要分成多少段
        in_channels=1,
        out_channels=1,
        hidden_size=1024,
        depth=24,
        heads=16,
        mlp_ratio=4.,
        drop_path=0.,
        use_heatmap=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # 计算1D patch的参数
        # 每行作为一个序列，每个patch的大小是width/patch_size
        self.tokens_per_row = patch_size  # 每行分成多少个token
        patch_dim = in_channels * (input_size // patch_size)  # 每个patch的特征维度
        
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # 时间步嵌入
        self.time_dim = hidden_size
        
        # 时间步编码
        self.time_embedding = nn.Sequential(
            nn.Linear(self.time_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.use_heatmap = use_heatmap
        self.heatmap_processor = HeatmapProcessor(hidden_size) if use_heatmap else None
        
        self.transformer = Transformer(
            dim=hidden_size,
            depth=depth,
            heads=heads,
            dim_head=hidden_size // heads,
            mlp_dim=int(hidden_size * mlp_ratio),
            dropout=drop_path,
            use_heatmap=use_heatmap
        )

        # 输出层：将token转回原始维度
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 8* out_channels)  # 每个patch的输出维度
        )
        
        # 复杂的行自适应模块
        if input_size==256:
            input_size = 160 
        adaptive_hidden = input_size*out_channels* 2  # 增加中间层维度
        self.row_adaptive = nn.Sequential(
            nn.LayerNorm(input_size*out_channels),  # 首先归一化输入
            nn.Linear(input_size*out_channels, adaptive_hidden),
            nn.GELU(),  # 非线性激活
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(adaptive_hidden, adaptive_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(adaptive_hidden, input_size*out_channels),
            nn.LayerNorm(input_size*out_channels)  # 最后再归一化
        )
        
        # 行注意力模块

        self.row_attention = nn.Sequential(
            nn.Linear(input_size*out_channels, input_size*out_channels),
            nn.Sigmoid()  # 生成注意力权重
        )

    def forward(self, x, time_emb, heatmap=None):
        b, c, h, w = x.shape
        #import pdb;pdb.set_trace()
        # 1D Patchify：每行分成patch_size个token
        # 原始形状: [batch, channel, height, width]
        # 将宽度维度分成patch_size份
        patch_width = w // self.tokens_per_row
        
        # 重排以进行1D patch化
        # 将每行分成patch_size个段，每段宽度为patch_width
        x = rearrange(x, 'b c h (n w) -> b (h n) (w c)', n=self.tokens_per_row, w=patch_width)
        x = self.to_patch_embedding(x)

        # 处理时间步编码
        t_emb = self.get_timestep_embedding(time_emb, self.time_dim)
        time_tokens = self.time_embedding(t_emb)
        x = x + time_tokens.unsqueeze(1)
        
        # Process heatmap if available
        condition = None
        if heatmap is not None and self.heatmap_processor is not None:
            condition = self.heatmap_processor(heatmap)
            
        # Apply transformer
        x = self.transformer(x, condition=condition)

        # 恢复原始形状
        x = self.to_pixels(x)
        x = rearrange(x, 'b (h n) (w c) -> b c h (n w)', 
                     h=h, n=self.tokens_per_row, 
                     w=patch_width, c=self.out_channels)
        
        # 对每一行应用复杂的自适应模块
        # 将形状从 [batch, channel, height, width] 转换为 [batch*channel*height, width]
        x = x.contiguous().reshape(b * c * h, -1)
        

        attention_weights = self.row_attention(x)
        
        # 应用自适应模块
        x_adapted = self.row_adaptive(x)
        
        # 应用注意力机制（元素级乘法）
        x_adapted = x_adapted * attention_weights
        
        # 添加残差连接
        x_adapted = x_adapted + x
        
        # 恢复原始形状
        x = x_adapted.view(b, c*self.out_channels//self.in_channels, h, -1)
        
        return x, None

import torch
import torch.nn as nn
import math
from abc import abstractmethod

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, **kwargs):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class ExpertMLP(nn.Module):
    """单个专家，使用MLP结构"""
    def __init__(self, in_channels, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels)
        )
    
    def forward(self, x):
        return self.net(x)

class HeatmapProcessor(nn.Module):
    """处理热力图信息以用于路由器的条件控制"""
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
        # 处理heatmap得到条件向量
        features = self.processor(heatmap)
        return torch.mean(features, dim=(2, 3))  # (B, hidden_size)

class Router(nn.Module):
    """基础路由器，支持可选的条件控制和预路由全局偏差校正"""
    def __init__(self, input_dim, num_experts, condition_dim=None, use_pre_routing_bias=False):
        super().__init__()
        self.num_experts = num_experts
        self.use_pre_routing_bias = use_pre_routing_bias
        
        # 基础路由层
        self.base_router = nn.Linear(input_dim, num_experts)
        
        # 条件调制参数（如果使用热力图）
        if condition_dim is not None:
            self.condition_scale = nn.Linear(condition_dim, num_experts)
            self.condition_shift = nn.Linear(condition_dim, num_experts)
        else:
            self.condition_scale = None
            self.condition_shift = None
            
        # 新增：预路由全局偏差校正网络
        if self.use_pre_routing_bias:
            self.global_bias_net = nn.Sequential(
                nn.Linear(num_experts, num_experts),
                nn.GELU(),
                nn.Linear(num_experts, num_experts)
            )

    def forward(self, x, condition=None):
        # 1. 计算初始路由分数（logits）
        initial_scores = self.base_router(x)
        
        final_scores = initial_scores

        # 2. 新增：应用预路由全局偏差校正
        if self.use_pre_routing_bias:
            # a. 聚合所有token的初始意向分布
            initial_distribution = torch.mean(initial_scores, dim=0) # [num_experts]
            # b. 生成全局校正偏差
            correction_bias = self.global_bias_net(initial_distribution) # [num_experts]
            # c. 应用校正
            final_scores = final_scores + correction_bias.unsqueeze(0) # 广播到 [N, num_experts]

        # 3. （保留原有逻辑）应用heatmap条件调制
        if condition is not None and self.condition_scale is not None:
            # 生成并应用调制参数
            scale = 1 + self.condition_scale(condition).unsqueeze(1)
            shift = self.condition_shift(condition).unsqueeze(1)
            final_scores = scale * final_scores + shift
        
        # 4. 使用softmax获得最终专家权重
        return torch.softmax(final_scores, dim=-1)

class MoEBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        num_experts=4,
        condition_dim=256,
        top_k=1,
        capacity_factor=None,
        use_pre_routing_bias=False,  # 新增参数
        **kwargs
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.num_experts = num_experts
        self.use_scale_shift_norm = use_scale_shift_norm
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.use_pre_routing_bias = use_pre_routing_bias # 保存参数

        # === emb -> 输入调控（FiLM/AdaGN）===
        if self.use_scale_shift_norm:
            self.in_norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
            self.emb_to_scale_shift = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, 2 * channels)
            )
        else:
            self.emb_to_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, channels)
            )

        # === 多个专家 ===
        self.experts = nn.ModuleList([
            ExpertMLP(channels, emb_channels, dropout)
            for _ in range(num_experts)
        ])

        # === heatmap -> router 调控 ===
        self.heatmap_processor = HeatmapProcessor(condition_dim)
        # 将新参数传递给Router
        self.router = Router(channels, num_experts, condition_dim, use_pre_routing_bias=self.use_pre_routing_bias)

        # === 输出与残差通道对齐 ===
        if self.channels != self.out_channels:
            self.proj = nn.Conv2d(channels, self.out_channels, 1)
            self.skip = nn.Conv2d(channels, self.out_channels, 1)
        else:
            self.proj = nn.Identity()
            self.skip = nn.Identity()

    @torch.no_grad()
    def _capacity_for(self, N_tokens):
        """计算每个专家的容量（若启用 capacity_factor）。"""
        if self.capacity_factor is None:
            return None
        base = math.ceil(N_tokens / self.num_experts)
        return max(1, int(math.ceil(self.capacity_factor * base)))

    def forward(self, x, emb, heatmap=None):
        B, C, H, W = x.shape
        N = B * H * W
        
        if self.use_scale_shift_norm:
            h = self.in_norm(x)
            scale, shift = self.emb_to_scale_shift(emb).chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        else:
            gate = torch.sigmoid(self.emb_to_gate(emb))
            h = x * gate[:, :, None, None]

        h_tokens = h.permute(0, 2, 3, 1).reshape(N, C)
        x_router_tokens = x.permute(0, 2, 3, 1).reshape(N, C)

        condition = None
        if heatmap is not None:
            cond_img = self.heatmap_processor(heatmap)
            condition = cond_img.repeat_interleave(H * W, dim=0)

        # 调用Router时无需再传递新参数，因为它已在初始化时配置好
        weights = self.router(x_router_tokens, condition)

        topk_vals, topk_idx = torch.topk(weights, k=self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        
        out_tokens = torch.zeros_like(h_tokens)
        per_expert_capacity = self._capacity_for(N)

        for e, expert in enumerate(self.experts):
            sel_mask = (topk_idx == e)
            positions = torch.nonzero(sel_mask, as_tuple=False)
            if positions.numel() == 0:
                continue

            token_ids = positions[:, 0]
            kpos = positions[:, 1]
            
            if per_expert_capacity is not None and token_ids.numel() > per_expert_capacity:
                with torch.no_grad():
                    w_e_full = topk_vals[token_ids, kpos]
                    top_sel = torch.topk(w_e_full, k=per_expert_capacity, dim=0).indices
                token_ids = token_ids[top_sel]
                kpos = kpos[top_sel]
            
            inputs_e = h_tokens[token_ids]
            w_e = topk_vals[token_ids, kpos].unsqueeze(1)
            out_e = expert(inputs_e)
            out_tokens.index_add_(0, token_ids, out_e * w_e)

        out = out_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        out = self.proj(out)
        return out + self.skip(x)
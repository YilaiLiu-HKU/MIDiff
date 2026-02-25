# smlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple

# --- 从您的项目中导入必要的模块 ---
from nn import (
    checkpoint,
    conv_nd, # 使用 conv_nd 保证维度一致性
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from fp16_util import convert_module_to_f16, convert_module_to_f32

# --- TimestepBlock 基类 ---
class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

# --- SparseAxialMLP (来自 unet.py, 适应动态 H/W) ---
class SparseAxialMLP(nn.Module):
    # ... (保持不变, 同上一个回答中的代码) ...
    def __init__(self, H: int, W: int, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        self.init_H, self.init_W = H, W
        h_mid = max(1, H // reduction); w_mid = max(1, W // reduction)
        self.h_fc1 = nn.Parameter(torch.randn(H, h_mid) * (2.0 / (H + h_mid))**0.5)
        self.h_fc2 = nn.Parameter(torch.randn(h_mid, H) * (2.0 / (h_mid + H))**0.5)
        self.w_fc1 = nn.Parameter(torch.randn(W, w_mid) * (2.0 / (W + w_mid))**0.5)
        self.w_fc2 = nn.Parameter(torch.randn(w_mid, W) * (2.0 / (w_mid + W))**0.5)
        self.act = nn.GELU(); self.drop = nn.Dropout(dropout)
        self.h_gate = nn.Parameter(torch.tensor(0.5)); self.w_gate = nn.Parameter(torch.tensor(0.5))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H == self.init_H: h_fc1_use, h_fc2_use = self.h_fc1, self.h_fc2
        else:
             h_fc1_use = F.interpolate(self.h_fc1.unsqueeze(0).permute(0, 2, 1), size=H, mode='linear', align_corners=False).permute(0, 2, 1).squeeze(0)
             h_fc2_use = F.interpolate(self.h_fc2.unsqueeze(0).unsqueeze(0), size=(self.h_fc2.shape[0], H), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        x_h = x.permute(0, 3, 1, 2).contiguous().view(B * W * C, H)
        try: x_h = x_h @ h_fc1_use; x_h = self.act(x_h); x_h = self.drop(x_h); x_h = x_h @ h_fc2_use
        except RuntimeError as e: print(f"Error H-mix: x_h {x_h.shape}, h1 {h_fc1_use.shape if h_fc1_use is not None else 'None'}, h2 {h_fc2_use.shape if h_fc2_use is not None else 'None'}. Err: {e}"); raise e
        x_h = x_h.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()
        if W == self.init_W: w_fc1_use, w_fc2_use = self.w_fc1, self.w_fc2
        else:
             w_fc1_use = F.interpolate(self.w_fc1.unsqueeze(0).permute(0, 2, 1), size=W, mode='linear', align_corners=False).permute(0, 2, 1).squeeze(0)
             w_fc2_use = F.interpolate(self.w_fc2.unsqueeze(0).unsqueeze(0), size=(self.w_fc2.shape[0], W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        x_w = x.permute(0, 2, 1, 3).contiguous().view(B * H * C, W)
        try: x_w = x_w @ w_fc1_use; x_w = self.act(x_w); x_w = self.drop(x_w); x_w = x_w @ w_fc2_use
        except RuntimeError as e: print(f"Error W-mix: x_w {x_w.shape}, w1 {w_fc1_use.shape if w_fc1_use is not None else 'None'}, w2 {w_fc2_use.shape if w_fc2_use is not None else 'None'}. Err: {e}"); raise e
        x_w = x_w.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()
        return self.h_gate.tanh() * x_h + self.w_gate.tanh() * x_w


# --- FeedForward (Channel MLP) ---
class FeedForward(nn.Module):
    # ... (保持不变) ...
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            normalization(dim), nn.Conv2d(dim, hidden_dim, 1), nn.GELU(),
            nn.Dropout(dropout), nn.Conv2d(hidden_dim, dim, 1), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

# --- Smlp基础块 (结合Token Mixing和Channel Mixing，并加入时间嵌入) ---
class SmlpBasicBlock(TimestepBlock):
    # ... (保持不变) ...
    def __init__(self, dim, H, W, emb_channels, mlp_ratio=4., dropout=0., drop_path=0., use_checkpoint=False):
        super().__init__()
        self.dim = dim; self.H = H; self.W = W; self.use_checkpoint = use_checkpoint
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * dim))
        self.norm1 = normalization(dim)
        self.token_mixer = SparseAxialMLP(H=H, W=W, reduction=4, dropout=dropout)
        # self.norm2 = normalization(dim) # Norm is now inside FeedForward
        hidden_dim = int(dim * mlp_ratio)
        self.channel_mixer = FeedForward(dim, hidden_dim, dropout=dropout)
        try: from timm.models.layers import DropPath; self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        except ImportError: print("Warn: timm.DropPath not found."); self.drop_path = nn.Identity()
    def _forward(self, x, emb):
        B, C, H, W = x.shape; assert C == self.dim; residual = x
        emb_out = self.emb_layers(emb).type(x.dtype);
        while len(emb_out.shape) < len(x.shape): emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.norm1(x); h = h * (1 + scale) + shift; h = self.token_mixer(h)
        x = residual + self.drop_path(h)
        h_channel = self.channel_mixer(x)
        x = x + self.drop_path(h_channel)
        return x
    def forward(self, x, emb):
        if self.use_checkpoint: return checkpoint(self._forward, (x, emb), self.parameters(), True)
        else: return self._forward(x, emb)


# --- Patch Embedding (处理非方形图像和 patch) ---
class PatchEmbed(nn.Module):
    # ... (保持不变) ...
    def __init__(self, img_size=(256, 160), patch_size=(4, 4), in_chans=1, embed_dim=128):
        super().__init__(); self.img_size = to_2tuple(img_size); self.patch_size = to_2tuple(patch_size)
        assert self.img_size[0] % self.patch_size[0] == 0 and self.img_size[1] % self.patch_size[1] == 0, f"Img dims {self.img_size} must be divisible by patch size {self.patch_size}."
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
    def forward(self, x): B, C, H, W = x.shape; assert H == self.img_size[0] and W == self.img_size[1], f"Input ({H}*{W}) != model ({self.img_size[0]}*{self.img_size[1]})."; return self.proj(x)

# --- Downsample Layer ---
class DownsampleLayer(nn.Module):
    # ... (保持不变) ...
     def __init__(self, in_dim, out_dim): super().__init__(); self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1); self.norm = normalization(out_dim)
     def forward(self, x): return self.norm(self.conv(x))

# --- Upsample Layer (新增) ---
class UpsampleLayer(nn.Module):
    """
    上采样层，使用 ConvTranspose2d。
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 使用 ConvTranspose2d 进行上采样
        self.conv_transpose = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = normalization(out_dim)

    def forward(self, x):
        return self.norm(self.conv_transpose(x))

# --- TimestepEmbedSequential (来自 unet.py 或 unet_new.py) ---
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    能够将时间步嵌入传递给子模块的 Sequential 容器。
    """
    def forward(self, x, emb, **kwargs): # 允许传递 kwargs
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb) # TimestepBlock 只接收 x 和 emb
            else:
                x = layer(x)
        return x


# --- 主模型 SmlpUnetModel (Encoder-Decoder Structure) ---
class SmlpUnetModel(nn.Module):
    """
    基于 Sparse MLP Block 构建的 UNet 结构扩散模型骨干。
    """
    def __init__(
        self,
        image_size=(256, 160),
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=2, # 用于控制每个分辨率下的 SmlpBasicBlock 数量
        channel_mult=(1, 2, 3, 4),
        learn_sigma=False,
        class_cond=False, # UNet 通常支持类别条件，但 SmlpBlock 当前不支持
        use_checkpoint=False,
        dropout=0.0,
        num_classes=None,
        use_fp16=False,
        patch_size=4,      # Patch 大小
        mlp_ratio=4.,
        drop_path_rate=0.1,
        dims=2,            # 保持 2D
        # --- UNet 特有参数 ---
        num_heads=-1, # Smlp 不直接用，但保留兼容性
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True, # 控制 SmlpBasicBlock 是否用 FiLM
        resblock_updown=False,    # 控制上/下采样方式 (这里我们用独立的 Layer)
        use_new_attention_order=False, # Smlp 不用
        attention_resolutions="", # Smlp 不直接用 attention_resolutions
        **kwargs
    ):
        super().__init__()
        image_size=(256, 160)
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.learn_sigma = learn_sigma
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_classes = num_classes # 保留但未使用

        # --- 时间步嵌入 ---
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        # --- 类别嵌入 (如果需要) ---
        if self.num_classes is not None:
             self.label_emb = nn.Embedding(num_classes, time_embed_dim)


        # --- Patch Embedding ---
        # UNet 通常在原始分辨率上操作，我们这里借鉴 PatchEmbed 但输出通道为 model_channels
        # 如果 patch_size > 1, 则输入层等效于 PatchEmbed
        # 如果 patch_size = 1, 则输入层只是一个普通的卷积
        initial_padding = (self.patch_size[0] // 2, self.patch_size[1] // 2) if self.patch_size != (1, 1) else 1
        self.input_conv = conv_nd(
            dims, in_channels, model_channels, kernel_size=self.patch_size,
            stride=self.patch_size, padding=0 # PatchEmbed 不用 padding
        )
        # self.input_conv = conv_nd(dims, in_channels, model_channels, 3, padding=1) # 标准 UNet 输入卷积


        # --- Encoder ---
        self.input_blocks = nn.ModuleList()
        input_block_chans = [model_channels] # 存储每个分辨率下的通道数，用于跳跃连接
        ch = model_channels
        current_H, current_W = self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1] # Patch embed 后的 H, W
        # current_H, current_W = self.image_size # 标准 UNet 输入 H, W
        ds = 1 # Downsampling factor
        #import pdb;pdb.set_trace()
        num_stages = len(channel_mult)
        # Calculate total number of SmlpBasicBlocks including middle and decoder
        total_encoder_blocks = num_res_blocks * num_stages
        total_decoder_blocks = (num_res_blocks + 1) * num_stages # Decoder has one extra block per stage for skip connection
        num_middle_blocks = 2 # Assuming 2 blocks in the middle
        total_blocks = total_encoder_blocks + total_decoder_blocks + num_middle_blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        block_idx = 0

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [
                    SmlpBasicBlock(
                        dim=ch, H=current_H, W=current_W, emb_channels=time_embed_dim,
                        mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr[block_idx],
                        use_checkpoint=use_checkpoint
                        # 注意：SmlpBasicBlock 没有 out_channels 参数，它保持通道数不变
                        # 如果需要改变通道数，需要在 SmlpBasicBlock 外部加层，或者修改 SmlpBasicBlock
                        # 为了简单起见，我们假设通道数在 stage 之间通过 Downsample/Upsample 改变
                        # 如果 SmlpBasicBlock 需要改变通道，需要修改 SmlpBasicBlock 本身
                    )
                ]
                # ch 保持不变，因为 SmlpBasicBlock 不改变通道
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
                block_idx += 1

            # --- Downsample (除了最后一个 stage) ---
            if level != len(channel_mult) - 1:
                downsample_layer = DownsampleLayer(ch, out_ch) # 使用 DownsampleLayer 改变通道数
                self.input_blocks.append(TimestepEmbedSequential(downsample_layer))
                ch = out_ch # 更新当前通道数
                input_block_chans.append(ch)
                current_H = (current_H + 1) // 2
                current_W = (current_W + 1) // 2
                ds *= 2

        # --- Middle Block ---
        self.middle_block = TimestepEmbedSequential(
            SmlpBasicBlock(ch, current_H, current_W, time_embed_dim, mlp_ratio, dropout, dpr[block_idx], use_checkpoint),
            # 可以再加一个 SmlpBasicBlock
            SmlpBasicBlock(ch, current_H, current_W, time_embed_dim, mlp_ratio, dropout, dpr[block_idx + 1], use_checkpoint)
        )
        block_idx += 2


        # --- Decoder ---
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1): # 每个 stage 多一个 block 用于处理 skip connection
                # Pop channels from encoder for skip connection
                ich = input_block_chans.pop()
                layers = [
                    SmlpBasicBlock(
                        dim=ch + ich, # 输入通道 = 上一层输出 + 跳跃连接
                        H=current_H, W=current_W, emb_channels=time_embed_dim,
                        mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr[block_idx],
                        use_checkpoint=use_checkpoint
                        # 这里输出通道数仍然是 ch + ich，需要一个额外的层来降维
                    )
                ]
                # 添加一个 1x1 卷积来将通道数调整回目标 out_ch
                layers.append(conv_nd(dims, ch + ich, out_ch, 1))
                ch = out_ch # 更新当前通道数

                # --- Upsample (除了第一个 decoder stage) ---
                if level and i == num_res_blocks: # 在每个 stage 的最后进行上采样
                    upsample_layer = UpsampleLayer(ch, ch) # 上采样层不改变通道数
                    layers.append(upsample_layer)
                    current_H *= 2
                    current_W *= 2
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                block_idx += 1
        
        # --- Final Layer ---
        self.out_norm = normalization(ch)
        self.out_silu = nn.SiLU()
        # 最后使用 zero_module 初始化
        # 输出通道为 self.out_channels (已考虑 learn_sigma)
        # kernel_size=self.patch_size, stride=self.patch_size, padding=0 for inverse patching
        # 使用 ConvTranspose2d 来恢复分辨率
        self.final_conv = nn.ConvTranspose2d(
            model_channels, # 输入通道是解码器最后输出的通道数 (应该是 model_channels)
            self.out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )
        # self.final_conv = zero_module(conv_nd(dims, model_channels, self.out_channels, 3, padding=1))


    def forward(self, x, timesteps, y=None, **kwargs):
        # 时间步和类别嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
             assert y is not None, "class-conditional model requires y targets"
             emb = emb + self.label_emb(y)

        # 输入卷积 / Patch Embedding
        h = self.input_conv(x.type(self.dtype))
        hs = [h] # 存储跳跃连接

        # Encoder
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        # Middle Block
        h = self.middle_block(h, emb)
        #print(f"Middle block output shape: {h.shape}")
        # Decoder
        for module in self.output_blocks:
            # 获取跳跃连接 (从 hs 的末尾弹出)
            skip = hs.pop()
            # 调整跳跃连接通道数以匹配当前 h (如果需要，但这里 SmlpBlock 输入处理了 concat)
            # print(f"h shape: {h.shape}, skip shape: {skip.shape}")
            # 处理可能的尺寸不匹配 (由于下采样/上采样中的奇偶问题)
            if h.shape[-2:] != skip.shape[-2:]:
                 h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = module(h, emb)
        #import pdb;pdb.set_trace()
        # Final Layer
        h = self.out_norm(h)
        h = self.out_silu(h)
        h = self.final_conv(h) # B, out_C, H, W
        
        # 裁剪输出以精确匹配原始尺寸 (ConvTranspose 可能导致尺寸略大)
        output_size_h, output_size_w = h.shape[-2], h.shape[-1]
        if output_size_h > self.image_size[0] or output_size_w > self.image_size[1]:
             h = h[..., :self.image_size[0], :self.image_size[1]]


        return h, None # 返回 (output, None)
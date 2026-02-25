from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from triple_attention import TripletAttention
from fp16_util import convert_module_to_f16, convert_module_to_f32
from nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
#ckpt=/home/yilai/projects/poster/NetDiffus/ckpt/tiff/ema_0.9999_058000.pt
# =================================================================
# ===== 在 unet.py 中添加以下两个新类 =====
# =================================================================
class SparseAxialMLP(nn.Module):
    """
    稀疏轴向 MLP：分别沿 H、W 两个轴做 token-mixing（全局建模），
    通过低秩瓶颈 (H -> H//r -> H, W -> W//r -> W) 减少参数量。
    - 权重对每个通道共享（与 C 无关），从而是“稀疏”的全局连接。
    - 需要固定的 H、W。
    """
    def __init__(self, H: int, W: int, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        assert H > 1 and W > 1
        self.H, self.W = H, W
        h_mid = max(1, H // reduction)
        w_mid = max(1, W // reduction)

        # 沿 H 轴的 mixing： (B, C, H, W) -> 对每个列（W 个）上的 H 序列做 MLP
        self.h_fc1 = nn.Parameter(th.randn(H, h_mid) * (2.0 / (H + h_mid))**0.5)
        self.h_fc2 = nn.Parameter(th.randn(h_mid, H) * (2.0 / (h_mid + H))**0.5)

        # 沿 W 轴的 mixing： (B, C, H, W) -> 对每个行（H 个）上的 W 序列做 MLP
        self.w_fc1 = nn.Parameter(th.randn(W, w_mid) * (2.0 / (W + w_mid))**0.5)
        self.w_fc2 = nn.Parameter(th.randn(w_mid, W) * (2.0 / (w_mid + W))**0.5)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # 可选：轴向比例门控，平衡两条支路
        self.h_gate = nn.Parameter(th.tensor(0.5))
        self.w_gate = nn.Parameter(th.tensor(0.5))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert H == self.H and W == self.W, "输入空间分辨率需与初始化时一致"

        # ---- H 轴 mixing ----
        x_h = x.permute(0, 3, 1, 2).contiguous().view(B * W * C, H)
        x_h = x_h @ self.h_fc1
        x_h = self.act(x_h)
        x_h = self.drop(x_h)
        x_h = x_h @ self.h_fc2
        x_h = x_h.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()

        # ---- W 轴 mixing ----
        x_w = x.permute(0, 2, 1, 3).contiguous().view(B * H * C, W)
        x_w = x_w @ self.w_fc1
        x_w = self.act(x_w)
        x_w = self.drop(x_w)
        x_w = x_w @ self.w_fc2
        x_w = x_w.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()

        out = self.h_gate.tanh() * x_h + self.w_gate.tanh() * x_w
        return out
class SparseMLPBlock(nn.Module):
    """
    一个改进版的 SparseMLP Block，增加了局部处理层。
    """
    def __init__(self, C: int, H: int, W: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        mid = C * expansion
        self.norm = nn.LayerNorm(C)

        self.pre_proj = nn.Conv2d(C, mid, kernel_size=1)
        
        # 新增：一个用于提取局部特征的深度卷积层
        self.local_conv = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid)
        
        self.sparse_mlp = SparseAxialMLP(H=H, W=W, reduction=4, dropout=dropout)
        self.post_proj = nn.Conv2d(mid, C, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        x_tok = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x_tok = self.norm(x_tok)
        x = x_tok.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = self.pre_proj(x)
        
        # 在全局混合之前，先通过局部卷积
        # 为卷积也增加一个残差连接，使其成为可选的特征增强
        x = self.local_conv(x) + x 
        
        x = self.sparse_mlp(x)
        x = self.post_proj(x)
        x = self.drop(x)

        return residual + x

class PixelRefiner(nn.Module):
    """
    使用 SparseMLP 模块独立微调每行像素值的模块。
    """
    def __init__(self, in_channels, H, W, depth=2, expansion=4, dropout=0.1):
        super().__init__()
        self.H = H
        self.W = W
        blocks = []
        for _ in range(depth):
            blocks.append(
                SparseMLPBlock(
                    C=in_channels, H=H, W=W, expansion=expansion, dropout=dropout
                )
            )
        
        # 添加一个最终的1x1卷积层
        final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 将其权重和偏置初始化为零
        with th.no_grad():
            final_conv.weight.zero_()
            final_conv.bias.zero_()
            
        blocks.append(final_conv)
        
        self.blocks = nn.Sequential(*blocks)


    def forward(self, x):
        return self.blocks(x)
class AntiAliasDownsample(nn.Module):
    """
    使用模糊核进行抗混叠处理的下采样模块。
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None, k=3):
        super().__init__()
        if dims != 2:
            raise NotImplementedError("AntiAliasDownsample currently only supports 2D.")
        
        self.channels = channels
        self.out_channels = out_channels or channels
        
        # 模糊层，用于抗混叠
        pad = (k - 1) // 2
        self.blur = nn.Conv2d(self.channels, self.channels, k, stride=1, padding=pad,
                              groups=self.channels, bias=False)
        with th.no_grad():
            # 初始化为均匀模糊核
            nn.init.constant_(self.blur.weight, 1.0 / (k * k))
            
        # 投影层，执行下采样
        self.proj = conv_nd(dims, self.channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # 先模糊，再进行带步长的卷积下采样
        return self.proj(self.blur(x))


class PixelShuffleUpsample(nn.Module):
    """
    使用 PixelShuffle 的上采样模块，通常效果更平滑。
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None, scale=2):
        super().__init__()
        if dims != 2:
            raise NotImplementedError("PixelShuffleUpsample currently only supports 2D.")
            
        self.channels = channels
        self.out_channels = out_channels or channels
        
        # 1x1卷积，将通道数扩展到 (scale*scale) 倍
        self.proj = conv_nd(dims, self.channels, self.out_channels * (scale ** 2), 1)
        self.ps = nn.PixelShuffle(scale)
        # 3x3卷积，用于平滑特征
        self.smooth = conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.ps(self.proj(x))
        return self.smooth(x)
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with configurable attention: 'origin', 'triple', 'both'.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        attention_type='origin',  # 'origin', 'triple', 'both'
        backbone_type='resnet',  # 'resnet' or 'dit'
        use_pixel_refiner=False,  # 是否在输出端使用 PixelRefiner
  # 'unet' or 'dit'
    ):
        super().__init__()
        assert attention_type in ('origin', 'triple', 'both'), \
            "attention_type must be 'origin', 'triple', or 'both'"
        self.attention_type = attention_type
        self.use_pixel_refiner = use_pixel_refiner
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        print(dims)
        print(in_channels)
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    # origin attention
                    if attention_type in ('origin', 'both'):
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_checkpoint=use_checkpoint,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )
                    # triple attention
                    if attention_type in ('triple', 'both'):
                        layers.append(TripletAttention(gate_channels=ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # middle block
        mid_layers = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ]
        if attention_type in ('origin', 'both'):
            mid_layers.append(
                AttentionBlock(
                    ch,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    use_checkpoint=use_checkpoint,
                    use_new_attention_order=use_new_attention_order,
                )
            )
        if attention_type in ('triple', 'both'):
            mid_layers.append(TripletAttention(gate_channels=ch))
        mid_layers.append(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self.middle_block = TimestepEmbedSequential(*mid_layers)
        self._feature_size += ch

        # output blocks
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    if attention_type in ('triple', 'both'):
                        layers.append(TripletAttention(gate_channels=ch))
                    if attention_type in ('origin', 'both'):
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_checkpoint=use_checkpoint,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )
                    
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        ) if resblock_updown else
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        actual_channels = self.model_channels * self.channel_mult[-1]
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        #self.max_value_head = PixelMaxPredictor(actual_channels, expansion=8, heads=8)
        #self.row_predictor = RowPredictor(self.out_channels, expansion=8, heads=8)
        if self.use_pixel_refiner:
            self.pixel_refiner = PixelRefiner(out_channels, image_size, 160)

    def forward(self, x, timesteps, y=None):
        
        assert (y is not None) == (self.num_classes is not None),"must specify y if and only if the model is class-conditional"
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)
        h = x.type(self.dtype)
        #import pdb;pdb.set_trace()
        for module in self.input_blocks:
            
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        #print(h.shape)
        #max_pred = self.max_value_head(h).squeeze(-1)
        #print(x.shape)
        #import pdb;pdb.set_trace()
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        #print(h.shape)
        #aux_out = self.aux_head_new(h)
        #row_predictions = self.row_predictor(out)
        if self.use_pixel_refiner:
            out = out + self.pixel_refiner(out)
        return out, None

class RowPredictor(nn.Module):
    """
    预测每一行的特征值的模块
    """
    def __init__(self, in_channels, expansion=4, heads=8):
        super().__init__()
        mid_channels = in_channels * expansion
        
        # 空间信息提取
        self.spatial_process = nn.Sequential(
            # 使用深度可分离卷积处理空间信息
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, mid_channels, 1),
            # 使用实例归一化代替层归一化
            nn.InstanceNorm2d(mid_channels),
            nn.GELU()
        )
        
        # 行注意力模块
        self.row_attention = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, (1, 3), padding=(0, 1), groups=mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.InstanceNorm2d(mid_channels),
            nn.GELU()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels // 4, mid_channels, 1),
            nn.Sigmoid()
        )
        
        # 多头自注意力处理每一行
        self.norm = nn.LayerNorm(mid_channels)  # 这个LayerNorm是正确的，因为它用在重塑后的特征上
        self.self_attention = nn.MultiheadAttention(mid_channels, heads, batch_first=True)
        
        # 最终的预测头
        self.pred_head = nn.Sequential(
            nn.Linear(mid_channels, mid_channels // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid_channels // 2, 1)
        )

        # 初始化参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 空间特征提取
        feat = self.spatial_process(x)
        
        # 2. 行注意力
        row_weights = self.row_attention(feat)
        feat = feat * row_weights
        
        # 3. 通道注意力
        channel_weights = self.channel_attention(feat)
        feat = feat * channel_weights
        
        # 4. 转换维度用于自注意力 [B, C, H, W] -> [B*H, W, C]
        feat = feat.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        feat = feat.reshape(B*H, W, -1)  # [B, H, W, C] -> [B*H, W, C]
        
        # 5. 应用自注意力
        feat = self.norm(feat)
        feat, _ = self.self_attention(feat, feat, feat)
        
        # 6. 池化得到每行的表示
        feat = feat.mean(dim=1)  # [B*H, W, C] -> [B*H, C]
        
        # 7. 预测每行的值
        row_predictions = self.pred_head(feat)  # [B*H, 1]
        
        # 8. 重塑回原始批次大小
        row_predictions = row_predictions.reshape(B, H)  # [B*H, 1] -> [B, H]
        return row_predictions


class PixelMaxPredictor(nn.Module):
    def __init__(self, in_channels, expansion=4, heads=8):
        super().__init__()
        mid = in_channels * expansion

        self.sep_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, mid, 1),
            nn.GELU()
        )

        # 通道注意力：权重和偏置都用小随机值
        self.eca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, mid, 1, groups=mid)
        )
        nn.init.constant_(self.eca[1].weight, 1e-2)
        nn.init.constant_(self.eca[1].bias,   0.0)

        self.norm = nn.LayerNorm(mid, eps=1e-3)
        self.attn = nn.MultiheadAttention(mid, heads, batch_first=True)

        self.to_score = nn.Conv2d(mid, 1, 1)
        # 关键：权重 kaiming，偏置 0
        nn.init.kaiming_normal_(self.to_score.weight, nonlinearity='linear')
        nn.init.zeros_(self.to_score.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.sep_conv(x)

        att = self.eca(feat)          # 去掉 Sigmoid，防止全 0
        feat = feat * att

        feat_flat = feat.flatten(2).transpose(1, 2)
        feat_flat = self.norm(feat_flat)
        feat_flat, _ = self.attn(feat_flat, feat_flat, feat_flat)
        feat = feat_flat.transpose(1, 2).view(B, -1, H, W)

        score = self.to_score(feat)   # 这里先不要 Sigmoid
        max_pred = score.view(B, -1).max(1, keepdim=True)[0]

        # 在训练早期打印看看是否还是 0
        if th.isnan(max_pred).any():
            print("NAN detected!")
        return max_pred
class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)

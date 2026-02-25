# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from Embed import DataEmbedding, DataEmbedding2, TokenEmbedding, get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_with_resolution, get_1d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid_with_resolution
import copy
import random
import torch.nn.functional as F
from embedding.profile_transformer import TransformerAutoEncoder

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        Embeds scalar timesteps into vector representations.
        MLP after sin embedding
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size1, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        self.attn_time = Attention(hidden_size1, num_heads=num_heads, qkv_bias=True, attn_drop=0, proj_drop=0,**block_kwargs)
        self.attn_feature = Attention(hidden_size1, num_heads=num_heads, qkv_bias=True, attn_drop=0, proj_drop=0,
                              **block_kwargs)
        self.cross = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=hidden_size1, nhead=num_heads),num_layers=1)
        self.norm2 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size1 * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size1, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size1, 6 * hidden_size1, bias=True)
        )
        self.hide = hidden_size1

    def forward(self, x, c):
        B, N, T, C = x.shape
        #shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x0 = gate_msa[:,0] * self.attn_time(modulate(self.norm1(x[:,0]), shift_msa[:,0], scale_msa[:,0]))
        x1 = gate_msa[:,1] * self.attn_feature(modulate(self.norm1(x[:,1]), shift_msa[:,1], scale_msa[:,1]))
        #x_cross = self.cross(x0,x1)
        x = x + torch.stack((x0, x1), dim=1)
        # x = x + gate_msa * self.attn_time(modulate(self.norm1(x), shift_msa, scale_msa).reshape(B*N, T, C)).reshape(B, N, T, C)
        #x = x + gate_msa_f * self.attn_feature(modulate(self.norm2(x), shift_msa_f, scale_msa_f).permute(0,3,2,1).reshape(B*C,T,N)).reshape(B, C, T, N).permute(0,3,2,1)
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.linear = nn.Linear(hidden_size,  out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        args = None,
        input_size=32,
        patch_size=2,
        in_channels=2 * 2 * 2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        user_profile_size=768
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        # self.patch_size = patch_size
        self.num_heads = num_heads
        self.args = args
        self.hidden_size = hidden_size
        #added_test-------------------------------------------------
        self.Embedding = DataEmbedding(1, self.hidden_size, args=self.args)

        self.Embedding_plus_mask = DataEmbedding2(2, 2*hidden_size, args=self.args)

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, hidden_size)
        )#原来H*W是32*32
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50, hidden_size)
        )
        # self.decoder_pos_embed_spatial = nn.Parameter(
        #     torch.zeros(1, 1024, hidden_size)
        # )
        # self.decoder_pos_embed_temporal = nn.Parameter(
        #     torch.zeros(1, 50,  hidden_size)
        # )

        #---------------------------------------------------

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.input_projection = Conv1d_with_init(2 * hidden_size, hidden_size, 1)
        #self.input_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.input_projection1 = Conv1d_with_init(2 * hidden_size, hidden_size, 1)
        #self.user_pro1 = nn.Linear(user_profile_size,hidden_size)
        #self.user_pro2 = nn.Linear(hidden_size, hidden_size)
        """self.user_layernorm = nn.LayerNorm(user_profile_size)
        ###
        self.user_layernorm2= nn.LayerNorm(user_profile_size//12)
        self.user_projection=nn.Linear(user_profile_size,7*user_profile_size)
        self.user_pro1 = nn.Linear(user_profile_size//12,4*hidden_size)
        self.user_pro2 = nn.Linear(4*hidden_size, hidden_size)"""
        self.profile_encoder=TransformerAutoEncoder(hidden_size,nhead=8, num_layers=2)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights_trivial()




    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        # torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
        # torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

        #     # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.weekday_embed.weight.data, std=0.02)

        w = self.Embedding.value_embedding.tokenConv.weight.data

        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize profile_encoder components
        # Initialize encoder weights
        for layer in self.profile_encoder.encoder.encoder.layers:
            # Initialize self-attention weights
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.constant_(layer.self_attn.in_proj_bias, 0)
            nn.init.constant_(layer.self_attn.out_proj.bias, 0)
            
            # Initialize feedforward weights
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.constant_(layer.linear1.bias, 0)
            nn.init.constant_(layer.linear2.bias, 0)
            
            # Initialize layer norm weights
            nn.init.constant_(layer.norm1.weight, 1.0)
            nn.init.constant_(layer.norm1.bias, 0)
            nn.init.constant_(layer.norm2.weight, 1.0)
            nn.init.constant_(layer.norm2.bias, 0)

        # Initialize decoder weights
        for layer in self.profile_encoder.decoder.decoder.layers:
            # Initialize self-attention weights
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.constant_(layer.self_attn.in_proj_bias, 0)
            nn.init.constant_(layer.self_attn.out_proj.bias, 0)
            
            # Initialize cross-attention weights
            nn.init.xavier_uniform_(layer.multihead_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.multihead_attn.out_proj.weight)
            nn.init.constant_(layer.multihead_attn.in_proj_bias, 0)
            nn.init.constant_(layer.multihead_attn.out_proj.bias, 0)
            
            # Initialize feedforward weights
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.constant_(layer.linear1.bias, 0)
            nn.init.constant_(layer.linear2.bias, 0)
            
            # Initialize layer norm weights
            nn.init.constant_(layer.norm1.weight, 1.0)
            nn.init.constant_(layer.norm1.bias, 0)
            nn.init.constant_(layer.norm2.weight, 1.0)
            nn.init.constant_(layer.norm2.bias, 0)
            nn.init.constant_(layer.norm3.weight, 1.0)
            nn.init.constant_(layer.norm3.bias, 0)

        # # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        # torch.nn.init.normal_(self.mask_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=0.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:  # Check if elementwise_affine is True
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)



    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        T, H, W = self.args.info
        # c = self.out_channels
        t = T//self.args.t_patch_size
        h = H // self.args.patch_size
        w = W // self.args.patch_size
        sigma_split = 2 if self.learn_sigma else 1


        x = x.reshape(x.shape[0],self.args.input_channels, t, h, w, self.args.t_patch_size, self.args.patch_size, self.args.patch_size,  sigma_split)
        # x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('ndthwabcs->ndsatbhcw', x)
        imgs = x.reshape(x.shape[0],self.args.input_channels, T,H, W)
        return imgs



        return  mask.float()

    def get_weights_sincos(self, num_t_patch, num_patch_1, num_patch_2):
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1 = num_patch_1,
            grid_size2 = num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
                torch.zeros(1, num_patch_1 * num_patch_2, self.hidden_size)
            )
        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.hidden_size)
        )

        pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(pos_embed_temporal.shape[-1], np.arange(num_t_patch, dtype=np.float32))

        pos_embed_temporal.data.copy_(torch.from_numpy(pos_temporal_emb).float().unsqueeze(0))

        pos_embed_spatial.requires_grad = False
        pos_embed_temporal.requires_grad = False

        return pos_embed_spatial, pos_embed_temporal, copy.deepcopy(pos_embed_spatial), copy.deepcopy(pos_embed_temporal)

    def pos_embed_enc(self, batch, input_size):

        if self.args.pos_emb == 'trivial':
            pos_embed = self.args.pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.args.pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.args.pos_emb == 'SinCos':
            pos_embed_spatial, pos_embed_temporal, _, _ = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])

            pos_embed = pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )
        pos_embed = pos_embed

        pos_embed = pos_embed.expand(batch, -1, -1)


        return pos_embed

    def forward(self, x, mask_origin, t, word,user_profile_emb, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels (时间戳)
        """
        #word 为word2vec embed的数据集名字
        #gaussian diffusion 798 行，input为cond和noise的concat
        N, imput_dim, T, H, W = x.shape

        N, C = word.shape


        TimeEmb = self.Embedding(x, y, is_time=True)
        T = T // self.args.t_patch_size
        input_size = (T, H // self.args.patch_size, W // self.args.patch_size)
        pos_embed_sort = self.pos_embed_enc( N, input_size)
        #####-----------------------------------------------------------------------####
        #word_emb = word.unsqueeze(1).repeat(1, TimeEmb.shape[1], 1).to(torch.float32)
        #####-----------------------------------------------------------------------####

        x_noise_mask = x[:,self.args.input_channels:]#noise
        x_obs = x[:,:self.args.input_channels]#条件
#-------------------------------------------------------------------
        #import pdb;pdb.set_trace()
        """user_profile_emb = self.user_layernorm(user_profile_emb)
        user_profile_emb=F.gelu(self.user_projection(user_profile_emb))
        channels=user_profile_emb.shape[1]
        user_profile_emb = user_profile_emb.reshape(-1, 84, channels // 84)
        user_profile_emb=self.user_pro1(self.user_layernorm2(user_profile_emb))
        user_profile_hidden = self.user_pro2(F.gelu(user_profile_emb))"""
        emb,emb_mask=user_profile_emb
        emb=self.profile_encoder(emb,emb_mask)
#-------------------------------------------------------------------

        x_mask_emb, obs_embed, mask_embed = self.Embedding_plus_mask(x_noise_mask, x_obs, mask_origin)

        _, L, _, C = x_mask_emb.shape#B，T，2(traffic+app),H*W
        # assert x_mask_emb.shape == pos_embed_sort.shape*2

        #x_mask_emb_comb = x_mask_emb + obs_embed

        # x_mask_emb = self.input_projection(torch.cat((x_mask_emb_comb, mask_embed), dim=3))
        #x_mask_emb = self.input_projection(torch.cat((x_mask_emb_comb, mask_embed), dim=3).permute(0, 2, 1)).permute(0, 2, 1)


        x_mask_emb_comb = x_mask_emb + obs_embed

        # x_mask_emb = self.input_projection(torch.cat((x_mask_emb_comb, mask_embed), dim=3))
        #过1*1层来project
        x_mask_emb1 = F.relu(self.input_projection(torch.cat((x_mask_emb_comb[:,0], mask_embed[:,0]), dim=2).permute(0, 2, 1)).permute(0, 2, 1))
        x_mask_emb2 = F.relu(self.input_projection1(torch.cat((x_mask_emb_comb[:,1], mask_embed[:,1]), dim=2).permute(0, 2, 1)).permute(0, 2, 1))
        x_mask_emb_com = torch.stack((x_mask_emb1, x_mask_emb2),dim=1)

        t = self.t_embedder(t)                   # (N, D)
        #transformer位置编码+x输入+扩散时间编码
        x_mask_emb =x_mask_emb_com + pos_embed_sort.to(device = t.device).unsqueeze(-1).repeat(1, 1, 1, self.args.input_channels).permute(0, 3,1,2)+ t.unsqueeze(1).unsqueeze(1)
        #import pdb;pdb.set_trace()
        # c =  torch.cat([word_emb, TimeEmb], dim =-1)
        #c = TimeEmb.unsqueeze(-1).repeat(1, 1, 1, 2).permute(0, 3,1,2) +user_profile_hidden.unsqueeze(-1).repeat(1, 1, 1, 2).permute(0, 3,1,2)
        c = TimeEmb.unsqueeze(-1).repeat(1, 1, 1, self.args.input_channels).permute(0, 3,1,2) +emb.unsqueeze(-1).repeat(1, 1, 1, self.args.input_channels).permute(0, 3,1,2)
        #c = self.layer_norm1(c)
        #####-----------------------------------------------------------------------####
        # x_mask_emb = torch.cat([word_emb, x_mask_emb], dim =-1)
        for block in self.blocks:
            x_mask_emb = block(x_mask_emb, c)                      # (N, T, D)
        # x_mask_emb = x_mask_emb + pos_embed_sort.to(device = t.device)+  t.unsqueeze(1)
        x = self.final_layer(x_mask_emb, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x, mask_origin

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

# def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
#     """
#     grid_size: int of the grid height and width
#     return:
#     pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#     """
#     grid_h = np.arange(grid_size, dtype=np.float32)
#     grid_w = np.arange(grid_size, dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#     grid = np.stack(grid, axis=0)
#
#     grid = grid.reshape([2, 1, grid_size, grid_size])
#     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
#     if cls_token and extra_tokens > 0:
#         pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
#     return pos_embed


# def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#     assert embed_dim % 2 == 0
#
#     # use half of dimensions to encode grid_h
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
#
#     emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
#     return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(args=None,**kwargs):
    return DiT(args = args,depth=8, hidden_size=256, patch_size=1, in_channels=1*1*1, num_heads=8,user_profile_size=768,  **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}



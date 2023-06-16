import torch
import torch.nn as nn
from itertools import repeat
from functools import reduce
import collections.abc
import math

import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Mlp, PatchEmbed # Attention

import torch.nn.functional as F
import random
# assert timm.__version__ == "0.3.2"  # version check
import einops
import torch.utils.checkpoint
import numpy as np

try:
    import xformers
    import xformers.ops
    ATTENTION = "xformer"
except:
    ATTENTION = "vanilla"

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    # x: B, N, C
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

        if ATTENTION == "xformer":
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)

            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)

        else:
           
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False,
                 dropout = 0.0, attn_dropout=0.0, proj_dropout=0.0,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop = attn_dropout, proj_drop=proj_dropout,
            )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint
        self.drop_out = nn.Dropout(dropout)

    # def forward(self, x, skip=None):
    #     if self.use_checkpoint:
    #         return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
    #     else:
    #         return self._forward(x, skip)

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_out( self.attn(self.norm1(x)) )
        x = x + self.drop_out( self.mlp(self.norm2(x)) )
        return x


class Decoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        # convert to final dimension

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num=0):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, :-return_token_num])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x


class UViTEncoder(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                    depth=12, num_heads=12, mlp_ratio=4.,
                    qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1, use_checkpoint=False,
                    dropout=0, attn_dropout = 0, proj_dropout = 0,

                    ):
        super().__init__()


        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.num_patches = self.patch_embed.num_patches
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.uncond_embed = nn.Parameter(torch.zeros(1, embed_dim))
            trunc_normal_(self.uncond_embed, std=.02)
            self.extras = 2
        else:
            self.extras = 1

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, dropout=dropout, attn_dropout = attn_dropout, proj_dropout = proj_dropout,
                )
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, dropout=dropout, attn_dropout = attn_dropout, proj_dropout = proj_dropout,)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=True, use_checkpoint=use_checkpoint, dropout=dropout, attn_dropout = attn_dropout, proj_dropout = proj_dropout,)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.num_patches, embed_dim))

        trunc_normal_(self.encoder_pos_embed, std=.02)
        self.apply(self._init_weights) # only initialize linear layer and layernorm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"encoder_pos_embed"}

    def forward(self, x, timesteps, mask=None, y=None):

        x = self.patch_embed(x)

        # time condition
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)

        # class condition
        if self.extras == 2:
            if y is None:
                # use unconditional embedding
                B, *_ = x.shape
                label_emb = self.uncond_embed.expand(B, -1, -1).type_as(x).to(x.device)
            else:
                label_emb = self.label_emb(y)
                label_emb = label_emb.unsqueeze(dim=1) # (1, 1, D）

            x = torch.cat((label_emb, x), dim=1)

        x = x + self.encoder_pos_embed

        B, N, C = x.shape
        # mask will skip time_token, etc
        if mask is not None:
            x = x[~mask].reshape(B, -1, C)

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)

        return x


class MaskedDWTEncoder(nn.Module):

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, data_chans=3,
                    depth=12, num_heads=12, mlp_ratio=4.,
                    qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1, use_checkpoint=False,

                    level = 1,
                    scale = 2,
                    ):
        super().__init__()

        self.level = level # level of DWT
        self.img_size = img_size
        self.patch_size = patch_size
        approx_chans = data_chans
        self.approx_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=approx_chans, embed_dim=embed_dim)

        self.scale = scale # scale factor of the patch embedding size of approximation to that of details
        details_chans = ( (level*2)**2 - 1 ) * approx_chans
        self.detail_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size*scale, in_chans=details_chans, embed_dim=embed_dim)

        self.approx_num_patches = self.approx_patch_embed.num_patches
        self.detail_num_patches = self.detail_patch_embed.num_patches
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=True, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)

        # two sets of patches: approximation and details
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.approx_num_patches + self.detail_num_patches, embed_dim))

        trunc_normal_(self.encoder_pos_embed, std=.02)
        self.apply(self._init_weights) # only initialize linear layer and layernorm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"encoder_pos_embed"}

    def forward(self, x, timesteps, mask=None,y=None):

        approx, detail = x[:, :3], x[:, 3:]

        approx = self.approx_patch_embed(approx)
        detail = self.detail_patch_embed(detail)
        x = torch.cat((approx, detail), dim=1)

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None or self.extras == 2:
            if y is None:
                y = torch.randint(0, self.num_classes, (x.shape[0],)).to(x.device)
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)

        x = x + self.encoder_pos_embed

        B, N, C = x.shape
        # mask will skip time_token, etc
        if mask is not None:
            # if training
            x = x[~mask].reshape(B, -1, C)

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        
        return x


class MaskedDWTUViT(nn.Module):

    def __init__(self, 
                img_size=224, patch_size=16, 
                in_chans=12, data_chans=3,
                encoder_embed_dim=768, encoder_depth=12, encoder_heads=12, 
                decoder_embed_dim=768, decoder_depth=12, decoder_heads=12,
                mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,

                num_classes=-1, 
                mlp_time_embed= False,
                use_checkpoint= False,

                level = 1, # NOTE: level is fixed to be 1
                scale = 2,

                **kwargs,
                ):
        super().__init__()


        self.name = "mask_dwt"
        self.num_features = self.encoder_embed_dim = encoder_embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.data_chans =data_chans
        self.level = 1 # NOTE: level is fixed to be 1
        self.scale = scale
        self.extras = 2 if self.num_classes > 0 else 1

        self.encoder = MaskedDWTEncoder(
                    img_size=img_size, patch_size=patch_size, data_chans=data_chans,
                    embed_dim=encoder_embed_dim, 
                    depth=encoder_depth,
                    num_heads=encoder_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                    mlp_time_embed=mlp_time_embed, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    level=level,
                    scale =scale,
        )

        self.approx_num_patches = self.encoder.approx_patch_embed.num_patches
        self.detail_num_patches = self.encoder.detail_patch_embed.num_patches
        self.patch_size = self.encoder.approx_patch_embed.patch_size[0] # effective patch size
        self.detail_patch_size = self.encoder.detail_patch_embed.patch_size[0]
        # self.patch_dim = self.patch_size ** 2 * in_chans
        self.approx_chans = self.data_chans
        approx_num_classes = self.patch_size ** 2 * data_chans
        self.detail_chans = ( (level*2)**2 - 1 ) * data_chans
        detail_num_classes = self.detail_patch_size ** 2 * self.detail_chans

        self.decoder_depth = decoder_depth
        if decoder_depth == 0:
            print("Decoder depth is zero, use two linear layers instead")
            self.approx_to_output = nn.Linear(encoder_embed_dim, approx_num_classes)
            self.detail_to_output = nn.Linear(encoder_embed_dim, detail_num_classes)
        else:
            print(f"Decoder: dim {decoder_embed_dim} depth {decoder_depth} head {decoder_heads}")
            self.approx_encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
            self.detail_encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim*2, bias=False)

            self.decoder_approx_pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.approx_num_patches, decoder_embed_dim))
            self.decoder_detail_pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.detail_num_patches, decoder_embed_dim*2))

            trunc_normal_(self.decoder_approx_pos_embed, std=.02)
            trunc_normal_(self.decoder_detail_pos_embed, std=.02)

            self.approx_to_output = Decoder(
                num_classes=approx_num_classes, 
                embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_heads, 
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, num_patches=self.approx_num_patches,
            )
            self.detail_to_output = Decoder(
                num_classes=detail_num_classes, 
                embed_dim=decoder_embed_dim*2, depth=decoder_depth, num_heads=decoder_heads*2, 
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, num_patches=self.detail_num_patches,
            )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"decoder_pos_embed"}

    def forward(self, x, timesteps, mask=None, y=None):

        # approx, detail = x
        # print(len(detail))
        # B, Cin, _, H, W = detail[0].shape
        # detail = detail[0].reshape(B, -1, H, W)

        x = self.encoder(x, timesteps, mask=mask, y=y)

        if mask is None:
            vis_approx_num = self.approx_num_patches
        else:
            vis_approx_num = (~mask[0, self.extras:self.extras+self.approx_num_patches]).sum()

        extras, approx, detail = x[:, :self.extras], x[:, self.extras:vis_approx_num+self.extras], x[:, vis_approx_num+self.extras:]

        if self.decoder_depth == 0:
            # use linear layer
            approx = self.approx_to_output(approx)
            detail = self.detail_to_output(detail)
        else:
            # use decoder
            approx = torch.cat((extras, approx), dim=1)
            detail = torch.cat((extras, detail), dim=1)

            # process approximate
            approx = self.approx_encoder_to_decoder(approx)
            B, N, C = approx.shape
            expand_pos_embed = self.decoder_approx_pos_embed.expand(B, -1, -1).type_as(approx).to(approx.device).clone().detach()
            if mask is not None:
                pos_emd_vis = expand_pos_embed[~mask[:, :self.extras+self.approx_num_patches]].reshape(B, -1, C)
                approx = approx + pos_emd_vis
            else:
                approx = approx + expand_pos_embed
            approx = self.approx_to_output(approx)
            approx = approx[:, self.extras:]

            # process detail
            detail = self.detail_encoder_to_decoder(detail)
            B, N, C = detail.shape
            expand_pos_embed = self.decoder_detail_pos_embed.expand(B, -1, -1).type_as(detail).to(detail.device).clone().detach()
            if mask is not None:
                pos_emd_vis = expand_pos_embed[torch.cat((~mask[:, :self.extras], ~mask[:, self.approx_num_patches+self.extras:]), dim=1)].reshape(B, -1, C)
                detail = detail + pos_emd_vis
            else:
                detail = detail + expand_pos_embed
            detail = self.detail_to_output(detail)
            detail = detail[:, self.extras:]

        if mask is None:
            # eval
            approx = unpatchify(approx, self.approx_chans) # b, c, h, w
            detail = unpatchify(detail, self.detail_chans) # b, c, h, w

            return torch.cat( (approx, detail), dim=1 )

        return approx, detail


class MaskedUViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                    encoder_embed_dim=768, 
                    decoder_embed_dim = 192,
                    decoder_depth = 4,
                    decoder_heads = 3,
    
                    encoder_depth=12, encoder_heads=12, mlp_ratio=4.,
                    qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,

                    num_classes=-1,
                    p_uncond = 0.15,

                    use_final_conv= False,
                    mlp_time_embed= False,
                    use_checkpoint= False,

                    dropout = 0,
                    attn_dropout = 0,
                    proj_dropout = 0,
                    **kwargs,
                    ):
        super().__init__()

        self.name = "mask_uvit"
        self.num_features = self.encoder_embed_dim = encoder_embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.p_uncond = p_uncond
        self.in_chans = in_chans

        # self.use_vit_decoder = use_vit_decoder
        self.use_final_conv = use_final_conv
        self.extras = 2 if self.num_classes > 0 else 1

        self.encoder = UViTEncoder(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                    embed_dim=encoder_embed_dim, 
                    depth=encoder_depth,
                    num_heads=encoder_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                    mlp_time_embed=mlp_time_embed, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    dropout = dropout, attn_dropout = 0, proj_dropout = 0,

        )

        num_patches = self.encoder.patch_embed.num_patches

        self.patch_size = self.encoder.patch_embed.patch_size[0] # effective patch size
        self.patch_dim = self.patch_size ** 2 * in_chans

        decoder_num_classes = self.patch_dim
        self.decoder_depth  = decoder_depth
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if decoder_depth != 0:
            self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, decoder_embed_dim))
            self.decoder =  Decoder(
                                num_classes = decoder_num_classes,
                                embed_dim=decoder_embed_dim, 
                                depth = decoder_depth,
                                num_heads = decoder_heads,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale, 
                                )
            trunc_normal_(self.decoder_pos_embed, std=.02)

        else:
            print("Decoder depth is zero, use linear layer instead")
            self.encoder_to_output = nn.Linear(encoder_embed_dim, decoder_num_classes)

        if use_final_conv:
            self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1)
        else:
            self.final_layer = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"decoder_pos_embed"}

    def forward(self, x, timesteps, mask=None, y=None):

        if self.num_classes > 0 and self.training and ( random.random() < self.p_uncond ):
            # train unconditionally
            y = None

        x = self.encoder(x, timesteps, mask=mask, y=y)

        if self.decoder_depth != 0:
            x = self.encoder_to_decoder(x)
            B, N, C = x.shape
            expand_pos_embed = self.decoder_pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

            if mask is not None:
                pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
                # pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
                # x = torch.cat([x + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
                x = x + pos_emd_vis
            else:
                x = x + expand_pos_embed

            noise = self.decoder(x, return_token_num=0) # shape (B, N+self.extras, C) return all tokens
        else:
            noise = self.encoder_to_output(x)

        noise = noise[:, self.extras:, :]
        if self.final_layer:
            noise = unpatchify(noise, self.in_chans)
            noise = self.final_layer(noise)

        elif mask is None:
            noise = unpatchify(noise, self.in_chans)
            return noise # when sampling no reconstruction is performed

        return noise




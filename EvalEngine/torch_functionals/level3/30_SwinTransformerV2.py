import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def mlp_fn(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, act_layer=nn.GELU(), drop_rate=0.):
    x = F.linear(x, fc1_weight, fc1_bias)
    x = act_layer(x)
    x = F.dropout(x, p=drop_rate)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.dropout(x, p=drop_rate)
    return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_attention_fn(x, mask, qkv_weight, q_bias, v_bias, logit_scale, cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight, proj_weight, proj_bias, relative_coords_table, relative_position_index, num_heads, attn_drop=0., proj_drop=0.):
    B_, N, C = x.shape
    qkv_bias = None
    if q_bias is not None:
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias)
    qkv = F.linear(input=x, weight=qkv_weight, bias=qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
    logit_scale = torch.clamp(logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()
    attn = attn * logit_scale

    cpb_mlp_output = F.linear(F.relu(F.linear(relative_coords_table, cpb_mlp_0_weight, cpb_mlp_0_bias)), cpb_mlp_2_weight)
    relative_position_bias_table = cpb_mlp_output.view(-1, num_heads)
    relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(
        relative_position_index.shape[0], relative_position_index.shape[1], -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
    else:
        attn = F.softmax(attn, dim=-1)

    attn = F.dropout(attn, p=attn_drop)
    x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=proj_drop)
    return x

def swin_transformer_block_fn(x, attn_mask, norm1_weight, norm1_bias, norm2_weight, norm2_bias, qkv_weight, q_bias, v_bias, logit_scale, cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight, proj_weight, proj_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, input_resolution, num_heads, window_size, shift_size, mlp_ratio=4., drop_path_rate=0., act_layer=nn.GELU()):
    H, W = input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = x.view(B, H, W, C)

    if shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x

    x_windows = window_partition(shifted_x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)

    attn_windows = window_attention_fn(
        x_windows, attn_mask, qkv_weight, q_bias, v_bias, logit_scale, 
        cpb_mlp_0_weight, cpb_mlp_0_bias, cpb_mlp_2_weight, 
        proj_weight, proj_bias, relative_coords_table, relative_position_index, num_heads
    )

    attn_windows = attn_windows.view(-1, window_size, window_size, C)
    shifted_x = window_reverse(attn_windows, window_size, H, W)

    if shift_size > 0:
        x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)
    x = F.layer_norm(x, (C,), norm1_weight, norm1_bias)
    x = shortcut + x

    x = x + mlp_fn(F.layer_norm(x, (C,), norm2_weight, norm2_bias), fc1_weight, fc1_bias, fc2_weight, fc2_bias, act_layer)
    return x

def patch_merging_fn(x, norm_weight, norm_bias, reduction_weight, input_resolution, dim):
    H, W = input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"
    assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

    x = x.view(B, H, W, C)
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], -1)
    x = x.view(B, -1, 4 * C)
    x = F.linear(x, reduction_weight)
    x = F.layer_norm(x, (x.size(-1), norm_weight, norm_bias)
    return x

def patch_embed_fn(x, proj_weight, proj_bias, norm_weight=None, norm_bias=None, img_size=(224, 224), patch_size=4):
    B, C, H, W = x.shape
    assert H == img_size[0] and W == img_size[1], f"Input image size ({H}*{W}) doesn't match model ({img_size[0]}*{img_size[1]})."
    x = F.conv2d(x, proj_weight, proj_bias, stride=patch_size)
    x = x.flatten(2).transpose(1, 2)
    if norm_weight is not None:
        x = F.layer_norm(x, (x.size(-1), norm_weight, norm_bias)
    return x

class Model(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Patch Embedding parameters
        self.proj_weight = nn.Parameter(torch.randn(embed_dim, in_chans, patch_size, patch_size))
        self.proj_bias = nn.Parameter(torch.zeros(embed_dim))
        if patch_norm:
            self.norm_weight = nn.Parameter(torch.ones(embed_dim))
            self.norm_bias = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.norm_weight = None
            self.norm_bias = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Layers parameters
        self.layers_params = nn.ParameterList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            for i_block in range(depths[i_layer]):
                # Norm1
                self.layers_params.append(nn.Parameter(torch.ones(dim)))
                self.layers_params.append(nn.Parameter(torch.zeros(dim)))
                # WindowAttention
                self.layers_params.append(nn.Parameter(torch.randn(dim * 3, dim)))
                if qkv_bias:
                    self.layers_params.append(nn.Parameter(torch.zeros(dim)))
                    self.layers_params.append(nn.Parameter(torch.zeros(dim)))
                else:
                    self.layers_params.append(None)
                    self.layers_params.append(None)
                self.layers_params.append(nn.Parameter(torch.log(10 * torch.ones((num_heads[i_layer], 1, 1)))))
                # CPB MLP
                self.layers_params.append(nn.Parameter(torch.randn(512, 2)))
                self.layers_params.append(nn.Parameter(torch.zeros(512)))
                self.layers_params.append(nn.Parameter(torch.randn(num_heads[i_layer], 512)))
                # Projection
                self.layers_params.append(nn.Parameter(torch.randn(dim, dim)))
                self.layers_params.append(nn.Parameter(torch.zeros(dim))))
                # Norm2
                self.layers_params.append(nn.Parameter(torch.ones(dim)))
                self.layers_params.append(nn.Parameter(torch.zeros(dim)))
                # MLP
                mlp_hidden_dim = int(dim * mlp_ratio)
                self.layers_params.append(nn.Parameter(torch.randn(mlp_hidden_dim, dim)))
                self.layers_params.append(nn.Parameter(torch.zeros(mlp_hidden_dim)))
                self.layers_params.append(nn.Parameter(torch.randn(dim, mlp_hidden_dim)))
                self.layers_params.append(nn.Parameter(torch.zeros(dim)))

            if i_layer < self.num_layers - 1:
                # PatchMerging
                self.layers_params.append(nn.Parameter(torch.ones(2 * dim)))
                self.layers_params.append(nn.Parameter(torch.zeros(2 * dim)))
                self.layers_params.append(nn.Parameter(torch.randn(2 * dim, 4 * dim)))

        # Final norm
        self.norm_weight = nn.Parameter(torch.ones(self.num_features))
        self.norm_bias = nn.Parameter(torch.zeros(self.num_features))

        # Head
        self.head_weight = nn.Parameter(torch.randn(num_classes, self.num_features))
        self.head_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, fn=None):
        if fn is None:
            fn = self.module_fn
        return fn(x, self)

    def module_fn(self, x, model):
        x = patch_embed_fn(
            x, 
            model.proj_weight, 
            model.proj_bias, 
            model.norm_weight if model.patch_norm else None, 
            model.norm_bias if model.patch_norm else None
        )
        x = model.pos_drop(x)

        param_idx = 0
        for i_layer in range(model.num_layers):
            dim = int(model.embed_dim * 2 ** i_layer)
            input_resolution = (model.patches_resolution[0] // (2 ** i_layer),
                               model.patches_resolution[1] // (2 ** i_layer))
            for i_block in range(model.depths[i_layer]):
                shift_size = 0 if (i_block % 2 == 0) else model.window_size // 2
                # Prepare attention mask if needed
                if shift_size > 0:
                    H, W = input_resolution
                    img_mask = torch.zeros((1, H, W, 1))
                    h_slices = (slice(0, -model.window_size),
                                slice(-model.window_size, -shift_size),
                                slice(-shift_size, None))
                    w_slices = (slice(0, -model.window_size),
                                slice(-model.window_size, -shift_size),
                                slice(-shift_size, None))
                    cnt = 0
                    for h in h_slices:
                        for w in w_slices:
                            img_mask[:, h, w, :] = cnt
                            cnt += 1
                    mask_windows = window_partition(img_mask, model.window_size)
                    mask_windows = mask_windows.view(-1, model.window_size * model.window_size)
                    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                else:
                    attn_mask = None

                # Get parameters
                norm1_weight = model.layers_params[param_idx]
                norm1_bias = model.layers_params[param_idx+1]
                qkv_weight = model.layers_params[param_idx+2]
                q_bias = model.layers_params[param_idx+3]
                v_bias = model.layers_params[param_idx+4]
                logit_scale = model.layers_params[param_idx+5]
                cpb_mlp_0_weight = model.layers_params[param_idx+6]
                cpb_mlp_0_bias = model.layers_params[param_idx+7]
                cpb_mlp_2_weight = model.layers_params[param_idx+8]
                proj_weight = model.layers_params[param_idx+9]
                proj_bias = model.layers_params[param_idx+10]
                norm2_weight = model.layers_params[param_idx+11]
                norm2_bias = model.layers_params[param_idx+12]
                fc1_weight = model.layers_params[param_idx+13]
                fc1_bias = model.layers_params[param_idx+14]
                fc2_weight = model.layers_params[param_idx+15]
                fc2_bias = model.layers_params[param_idx+16]
                param_idx += 17

                x = swin_transformer_block_fn(
                    x, attn_mask, norm1_weight, norm1_bias, norm2_weight, norm2_bias,
                    qkv_weight, q_bias, v_bias, logit_scale, cpb_mlp_0_weight, cpb_mlp_0_bias,
                    cpb_mlp_2_weight, proj_weight, proj_bias, fc1_weight, fc1_bias,
                    fc2_weight, fc2_bias, input_resolution, model.num_heads[i_layer],
                    model.window_size, shift_size, model.mlp_ratio, model.drop_path_rate
                )

            if i_layer < model.num_layers - 1:
                norm_weight = model.layers_params[param_idx]
                norm_bias = model.layers_params[param_idx+1]
                reduction_weight = model.layers_params[param_idx+2]
                param_idx += 3
                x = patch_merging_fn(x, norm_weight, norm_bias, reduction_weight, input_resolution, dim)

        x = F.layer_norm(x, (x.size(-1), model.norm_weight, model.norm_bias)
        x = x.transpose(1, 2).mean(dim=-1)
        x
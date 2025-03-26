import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc

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

def mlp_fn(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, act_layer=nn.GELU, drop_rate=0.):
    x = F.linear(x, fc1_weight, fc1_bias)
    x = act_layer()(x)
    x = F.dropout(x, p=drop_rate, training=True)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.dropout(x, p=drop_rate, training=True)
    return x

def swin_mlp_block_fn(x, norm1_weight, norm1_bias, spatial_mlp_weight, spatial_mlp_bias, norm2_weight, norm2_bias, 
                      fc1_weight, fc1_bias, fc2_weight, fc2_bias, input_resolution, num_heads, window_size, shift_size, 
                      mlp_ratio, drop_rate, drop_path_rate, act_layer=nn.GELU):
    H, W = input_resolution
    B, L, C = x.shape
    shortcut = x
    
    # norm1
    x = F.layer_norm(x, (C,), norm1_weight, norm1_bias)
    x = x.view(B, H, W, C)
    
    # shift
    if shift_size > 0:
        padding = [window_size - shift_size, shift_size, window_size - shift_size, shift_size]
        shifted_x = F.pad(x, [0, 0, padding[0], padding[1], padding[2], padding[3]], "constant", 0)
    else:
        shifted_x = x
    _, _H, _W, _ = shifted_x.shape
    
    # window partition
    x_windows = window_partition(shifted_x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)
    
    # spatial mlp
    x_windows_heads = x_windows.view(-1, window_size * window_size, num_heads, C // num_heads)
    x_windows_heads = x_windows_heads.transpose(1, 2)
    x_windows_heads = x_windows_heads.reshape(-1, num_heads * window_size * window_size, C // num_heads)
    spatial_mlp_windows = F.conv1d(x_windows_heads, spatial_mlp_weight, spatial_mlp_bias, groups=num_heads)
    spatial_mlp_windows = spatial_mlp_windows.view(-1, num_heads, window_size * window_size, C // num_heads).transpose(1, 2)
    spatial_mlp_windows = spatial_mlp_windows.reshape(-1, window_size * window_size, C)
    
    # merge windows
    spatial_mlp_windows = spatial_mlp_windows.reshape(-1, window_size, window_size, C)
    shifted_x = window_reverse(spatial_mlp_windows, window_size, _H, _W)
    
    # reverse shift
    if shift_size > 0:
        x = shifted_x[:, padding[2]:-_H+padding[3], padding[0]:-_W+padding[1], :].contiguous()
    else:
        x = shifted_x
    x = x.view(B, H * W, C)
    
    # FFN
    x = shortcut + x
    x = x + mlp_fn(F.layer_norm(x, (C,), norm2_weight, norm2_bias), fc1_weight, fc1_bias, fc2_weight, fc2_bias, act_layer, drop_rate)
    return x

def patch_merging_fn(x, norm_weight, norm_bias, reduction_weight, reduction_bias, input_resolution, dim):
    H, W = input_resolution
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], -1)
    x = x.view(B, -1, 4 * C)
    
    x = F.layer_norm(x, (4 * C,), norm_weight, norm_bias)
    x = F.linear(x, reduction_weight, reduction_bias)
    return x

def basic_layer_fn(x, params, input_resolution, depth, num_heads, window_size, mlp_ratio, drop_rate, drop_path_rate, downsample):
    for i in range(depth):
        x = swin_mlp_block_fn(
            x, 
            params[f'norm1_weight_{i}'], params[f'norm1_bias_{i}'],
            params[f'spatial_mlp_weight_{i}'], params[f'spatial_mlp_bias_{i}'],
            params[f'norm2_weight_{i}'], params[f'norm2_bias_{i}'],
            params[f'fc1_weight_{i}'], params[f'fc1_bias_{i}'],
            params[f'fc2_weight_{i}'], params[f'fc2_bias_{i}'],
            input_resolution, num_heads, window_size, 0 if (i % 2 == 0) else window_size // 2,
            mlp_ratio, drop_rate, drop_path_rate
        )
    if downsample:
        x = patch_merging_fn(
            x,
            params['downsample_norm_weight'], params['downsample_norm_bias'],
            params['downsample_reduction_weight'], params['downsample_reduction_bias'],
            input_resolution, params['dim']
        )
    return x

def patch_embed_fn(x, proj_weight, proj_bias, norm_weight=None, norm_bias=None, img_size=(224, 224), patch_size=(4, 4)):
    B, C, H, W = x.shape
    x = F.conv2d(x, proj_weight, proj_bias, stride=patch_size)
    x = x.flatten(2).transpose(1, 2)
    if norm_weight is not None:
        x = F.layer_norm(x, (x.size(-1),), norm_weight, norm_bias)
    return x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class Model(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution

        # Patch embedding parameters
        self.proj_weight = nn.Parameter(torch.randn(embed_dim, in_chans, patch_size[0], patch_size[1]))
        self.proj_bias = nn.Parameter(torch.zeros(embed_dim))
        if patch_norm:
            self.norm_weight = nn.Parameter(torch.ones(embed_dim))
            self.norm_bias = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.register_parameter('norm_weight', None)
            self.register_parameter('norm_bias', None)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Layer parameters
        self.layer_params = nn.ParameterDict()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            for i_block in range(depths[i_layer]):
                # Norm1
                self.layer_params[f'norm1_weight_{i_block}'] = nn.Parameter(torch.ones(dim))
                self.layer_params[f'norm1_bias_{i_block}'] = nn.Parameter(torch.zeros(dim))
                # Spatial MLP
                spatial_mlp_dim = num_heads[i_layer] * window_size ** 2
                self.layer_params[f'spatial_mlp_weight_{i_block}'] = nn.Parameter(torch.randn(spatial_mlp_dim, spatial_mlp_dim, 1))
                self.layer_params[f'spatial_mlp_bias_{i_block}'] = nn.Parameter(torch.zeros(spatial_mlp_dim))
                # Norm2
                self.layer_params[f'norm2_weight_{i_block}'] = nn.Parameter(torch.ones(dim))
                self.layer_params[f'norm2_bias_{i_block}'] = nn.Parameter(torch.zeros(dim))
                # MLP
                mlp_hidden_dim = int(dim * mlp_ratio)
                self.layer_params[f'fc1_weight_{i_block}'] = nn.Parameter(torch.randn(mlp_hidden_dim, dim))
                self.layer_params[f'fc1_bias_{i_block}'] = nn.Parameter(torch.zeros(mlp_hidden_dim))
                self.layer_params[f'fc2_weight_{i_block}'] = nn.Parameter(torch.randn(dim, mlp_hidden_dim))
                self.layer_params[f'fc2_bias_{i_block}'] = nn.Parameter(torch.zeros(dim))
            
            if i_layer < self.num_layers - 1:
                # Downsample
                self.layer_params[f'downsample_norm_weight'] = nn.Parameter(torch.ones(4 * dim))
                self.layer_params[f'downsample_norm_bias'] = nn.Parameter(torch.zeros(4 * dim))
                self.layer_params[f'downsample_reduction_weight'] = nn.Parameter(torch.randn(2 * dim, 4 * dim))
                self.layer_params[f'downsample_reduction_bias'] = nn.Parameter(torch.zeros(2 * dim))
                self.layer_params['dim'] = nn.Parameter(torch.tensor(dim), requires_grad=False)

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
        # Patch embed
        x = patch_embed_fn(
            x, 
            model.proj_weight, model.proj_bias, 
            model.norm_weight, model.norm_bias,
            model.patches_resolution, model.patch_size
        )
        x = F.dropout(x, p=model.drop_rate, training=self.training)

        # Layers
        for i_layer in range(model.num_layers):
            dim = int(model.embed_dim * 2 ** i_layer)
            input_resolution = (
                model.patches_resolution[0] // (2 ** i_layer),
                model.patches_resolution[1] // (2 ** i_layer)
            )
            x = basic_layer_fn(
                x,
                model.layer_params,
                input_resolution,
                model.depths[i_layer],
                model.num_heads[i_layer],
                model.window_size,
                model.mlp_ratio,
                model.drop_rate,
                model.drop_path_rate,
                i_layer < model.num_layers - 1
            )

        # Final norm and head
        x = F.layer_norm(x, (model.num_features,), model.norm_weight, model.norm_bias)
        x = x.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = F.linear(x, model.head_weight, model.head_bias)
        return x

batch_size = 10
image_size = 224

def get_inputs():
    return [torch.randn(batch_size, 3, image_size, image_size)]

def get_init_inputs():
    return []
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat
from typing import Optional, Tuple, Union, List, Any


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def mlp_forward(
    x: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    drop: float = 0.0,
) -> torch.Tensor:
    x = F.linear(x, fc1_weight, fc1_bias)
    x = F.gelu(x)
    x = F.dropout(x, p=drop, training=True)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.dropout(x, p=drop, training=True)
    return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_attention_forward(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_bias: Optional[torch.Tensor],
    v_bias: Optional[torch.Tensor],
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    logit_scale: torch.Tensor,
    cpb_mlp_weights: List[torch.Tensor],
    cpb_mlp_biases: List[torch.Tensor],
    relative_coords_table: torch.Tensor,
    relative_position_index: torch.Tensor,
    window_size: Tuple[int, int],
    num_heads: int,
    mask: Optional[torch.Tensor] = None,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
) -> torch.Tensor:
    B_, N, C = x.shape

    qkv_bias = None
    if q_bias is not None and v_bias is not None:
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias)
        )

    qkv = F.linear(input=x, weight=qkv_weight, bias=qkv_bias)
    qkv = qkv.reshape(B_, N, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # cosine attention
    attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
    logit_scale_clamped = torch.clamp(
        logit_scale, max=torch.log(torch.tensor(1.0 / 0.01, device=x.device))
    ).exp()
    attn = attn * logit_scale_clamped

    # Apply MLP to get position bias table
    x_pos = relative_coords_table
    for i in range(len(cpb_mlp_weights)):
        if i < len(cpb_mlp_weights) - 1:
            x_pos = F.relu(F.linear(x_pos, cpb_mlp_weights[i], cpb_mlp_biases[i]))
        else:
            x_pos = F.linear(x_pos, cpb_mlp_weights[i], cpb_mlp_biases[i])

    relative_position_bias_table = x_pos.view(-1, num_heads)
    relative_position_bias = relative_position_bias_table[
        relative_position_index.int().view(-1)
    ].view(window_size[0] * window_size[1], window_size[0] * window_size[1], -1)
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

    attn = F.dropout(attn, p=attn_drop, training=True)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=proj_drop, training=True)

    return x


def swin_transformer_block_forward(
    x: torch.Tensor,
    param_dict: dict,
    buffer_dict: dict,
    i_layer: int,
    i_block: int,
) -> torch.Tensor:
    # Define prefix for parameter and buffer keys
    prefix = f"layer{i_layer}_block{i_block}"

    # Retrieve normalization parameters
    norm1_weight = param_dict[f"norm1_weight_{prefix}"]
    norm1_bias = param_dict[f"norm1_bias_{prefix}"]
    norm2_weight = param_dict[f"norm2_weight_{prefix}"]
    norm2_bias = param_dict[f"norm2_bias_{prefix}"]

    # Retrieve buffers
    input_resolution = tuple(buffer_dict[f"input_resolution_layer{i_layer}"].tolist())
    window_size = int(buffer_dict["window_size"])
    shift_size = int(buffer_dict[f"shift_size_{prefix}"])
    attn_mask = buffer_dict.get(f"attn_mask_{prefix}", None)
    drop_path_rate = float(buffer_dict[f"drop_path_rate_{prefix}"])

    # Retrieve attention parameters
    qkv_weight = param_dict[f"qkv_weight_{prefix}"]
    q_bias = param_dict.get(f"q_bias_{prefix}", None)
    v_bias = param_dict.get(f"v_bias_{prefix}", None)
    proj_weight = param_dict[f"proj_weight_{prefix}"]
    proj_bias = param_dict[f"proj_bias_{prefix}"]
    logit_scale = param_dict[f"logit_scale_{prefix}"]
    cpb_mlp_weights = [
        param_dict[f"cpb_mlp_weight0_{prefix}"],
        param_dict[f"cpb_mlp_weight1_{prefix}"],
    ]
    cpb_mlp_biases = [
        param_dict[f"cpb_mlp_bias0_{prefix}"],
        None,
    ]
    relative_coords_table = buffer_dict[f"relative_coords_table_{prefix}"]
    relative_position_index = buffer_dict[f"relative_position_index_{prefix}"]
    num_heads = int(buffer_dict["num_heads"][i_layer])
    attn_drop = float(buffer_dict["attn_drop_rate"])
    proj_drop = float(buffer_dict["drop_rate"])

    # Retrieve MLP parameters
    fc1_weight = param_dict[f"fc1_weight_{prefix}"]
    fc1_bias = param_dict[f"fc1_bias_{prefix}"]
    fc2_weight = param_dict[f"fc2_weight_{prefix}"]
    fc2_bias = param_dict[f"fc2_bias_{prefix}"]
    drop = float(buffer_dict["drop_rate"])

    # Original implementation
    H, W = input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = F.layer_norm(x, (C,), weight=norm1_weight, bias=norm1_bias)
    x = x.view(B, H, W, C)

    if shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    else:
        shifted_x = x

    x_windows = window_partition(shifted_x, window_size)
    x_windows = x_windows.view(-1, window_size * window_size, C)

    attn_windows = window_attention_forward(
        x_windows,
        qkv_weight=qkv_weight,
        q_bias=q_bias,
        v_bias=v_bias,
        proj_weight=proj_weight,
        proj_bias=proj_bias,
        logit_scale=logit_scale,
        cpb_mlp_weights=cpb_mlp_weights,
        cpb_mlp_biases=cpb_mlp_biases,
        relative_coords_table=relative_coords_table,
        relative_position_index=relative_position_index,
        window_size=to_2tuple(window_size),
        num_heads=num_heads,
        mask=attn_mask,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
    )

    attn_windows = attn_windows.view(-1, window_size, window_size, C)
    shifted_x = window_reverse(attn_windows, window_size, H, W)

    if shift_size > 0:
        x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    if drop_path_rate > 0.0:
        keep_prob = 1.0 - drop_path_rate
        mask = torch.rand(B, 1, 1, device=x.device) >= drop_path_rate
        x = x / keep_prob * mask

    x = shortcut + x

    shortcut = x
    x = F.layer_norm(x, (C,), weight=norm2_weight, bias=norm2_bias)
    x = mlp_forward(
        x,
        fc1_weight=fc1_weight,
        fc1_bias=fc1_bias,
        fc2_weight=fc2_weight,
        fc2_bias=fc2_bias,
        drop=drop,
    )

    if drop_path_rate > 0.0:
        keep_prob = 1.0 - drop_path_rate
        mask = torch.rand(B, 1, 1, device=x.device) >= drop_path_rate
        x = x / keep_prob * mask

    x = shortcut + x

    return x


def patch_merging_forward(
    x: torch.Tensor,
    param_dict: dict,
    buffer_dict: dict,
    i_layer: int,
) -> torch.Tensor:
    # Retrieve parameters and buffers
    reduction_weight = param_dict[f"reduction_weight_layer{i_layer}"]
    norm_weight = param_dict[f"norm_weight_layer{i_layer}"]
    norm_bias = param_dict[f"norm_bias_layer{i_layer}"]
    input_resolution = tuple(buffer_dict[f"input_resolution_layer{i_layer}"].tolist())

    # Original implementation
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
    x = F.layer_norm(x, (x.size(-1),), weight=norm_weight, bias=norm_bias)

    return x


def basic_layer_forward(
    x: torch.Tensor,
    param_dict: dict,
    buffer_dict: dict,
    i_layer: int,
) -> torch.Tensor:
    # Get the number of blocks in this layer
    depths = buffer_dict["depths"].tolist()
    num_blocks = depths[i_layer]

    # Process each block
    for i_block in range(num_blocks):
        x = swin_transformer_block_forward(x, param_dict, buffer_dict, i_layer, i_block)

    # Apply downsampling if not the last layer
    if i_layer < int(buffer_dict["num_layers"]) - 1:
        x = patch_merging_forward(x, param_dict, buffer_dict, i_layer)

    return x


def patch_embed_forward(
    x: torch.Tensor,
    param_dict: dict,
    buffer_dict: dict,
) -> torch.Tensor:
    # Retrieve parameters and buffers
    proj_weight = param_dict["patch_embed_proj"]
    proj_bias = param_dict["patch_embed_bias"]
    patch_size = tuple(buffer_dict["patch_size"].tolist())
    norm_weight = param_dict.get("patch_embed_norm_weight", None)
    norm_bias = param_dict.get("patch_embed_norm_bias", None)

    # Original implementation
    B, C, H, W = x.shape
    x = F.conv2d(x, proj_weight, proj_bias, stride=patch_size)
    x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
    if norm_weight is not None and norm_bias is not None:
        x = F.layer_norm(x, (x.size(-1),), weight=norm_weight, bias=norm_bias)
    return x


class Model(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        **kwargs,
    ):
        super().__init__()

        ### Register Hyperparameters as Buffers ###
        self.register_buffer("num_layers", torch.tensor(len(depths), dtype=torch.long))
        self.register_buffer("embed_dim", torch.tensor(embed_dim, dtype=torch.long))
        patch_size_tuple = to_2tuple(patch_size)
        self.register_buffer(
            "patch_size", torch.tensor(patch_size_tuple, dtype=torch.long)
        )
        img_size_tuple = to_2tuple(img_size)
        self.register_buffer("img_size", torch.tensor(img_size_tuple, dtype=torch.long))
        self.register_buffer("in_chans", torch.tensor(in_chans, dtype=torch.long))
        self.register_buffer("num_classes", torch.tensor(num_classes, dtype=torch.long))
        self.register_buffer("window_size", torch.tensor(window_size, dtype=torch.long))
        self.register_buffer("mlp_ratio", torch.tensor(mlp_ratio, dtype=torch.float))
        self.register_buffer("qkv_bias", torch.tensor(qkv_bias, dtype=torch.bool))
        self.register_buffer("drop_rate", torch.tensor(drop_rate, dtype=torch.float))
        self.register_buffer(
            "attn_drop_rate", torch.tensor(attn_drop_rate, dtype=torch.float)
        )
        self.register_buffer(
            "drop_path_rate", torch.tensor(drop_path_rate, dtype=torch.float)
        )
        self.register_buffer("patch_norm", torch.tensor(patch_norm, dtype=torch.bool))
        self.register_buffer(
            "use_checkpoint", torch.tensor(use_checkpoint, dtype=torch.bool)
        )
        self.register_buffer(
            "pretrained_window_sizes",
            torch.tensor(pretrained_window_sizes, dtype=torch.long),
        )
        self.register_buffer("depths", torch.tensor(depths, dtype=torch.long))
        self.register_buffer("num_heads", torch.tensor(num_heads, dtype=torch.long))

        ### Compute and Register Additional Buffers ###
        patches_resolution = [
            img_size_tuple[0] // patch_size_tuple[0],
            img_size_tuple[1] // patch_size_tuple[1],
        ]
        self.register_buffer(
            "patches_resolution", torch.tensor(patches_resolution, dtype=torch.long)
        )
        self.register_buffer(
            "num_patches",
            torch.tensor(
                patches_resolution[0] * patches_resolution[1], dtype=torch.long
            ),
        )
        num_features = int(embed_dim * 2 ** (len(depths) - 1))
        self.register_buffer(
            "num_features", torch.tensor(num_features, dtype=torch.long)
        )

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.register_buffer("dpr", torch.tensor(dpr, dtype=torch.float))

        ### Patch Embedding Parameters ###
        self.register_parameter(
            "patch_embed_proj",
            nn.Parameter(
                torch.zeros(
                    embed_dim, in_chans, patch_size_tuple[0], patch_size_tuple[1]
                )
            ),
        )
        self.register_parameter(
            "patch_embed_bias", nn.Parameter(torch.zeros(embed_dim))
        )
        if patch_norm:
            self.register_parameter(
                "patch_embed_norm_weight", nn.Parameter(torch.ones(embed_dim))
            )
            self.register_parameter(
                "patch_embed_norm_bias", nn.Parameter(torch.zeros(embed_dim))
            )
        else:
            self.register_parameter("patch_embed_norm_weight", None)
            self.register_parameter("patch_embed_norm_bias", None)

        ### Layer Parameters and Buffers ###
        dpr_idx = 0  # Index for dpr
        for i_layer in range(len(depths)):
            current_dim = int(embed_dim * 2**i_layer)
            layer_input_resolution = [
                patches_resolution[0] // (2**i_layer),
                patches_resolution[1] // (2**i_layer),
            ]
            self.register_buffer(
                f"input_resolution_layer{i_layer}",
                torch.tensor(layer_input_resolution, dtype=torch.long),
            )

            # Blocks
            for i_block in range(depths[i_layer]):
                # Shift size
                shift_size = 0 if (i_block % 2 == 0) else window_size // 2
                self.register_buffer(
                    f"shift_size_layer{i_layer}_block{i_block}",
                    torch.tensor(shift_size, dtype=torch.long),
                )

                # Attention mask
                if shift_size > 0:
                    H, W = layer_input_resolution
                    img_mask = torch.zeros((1, H, W, 1))
                    h_slices = (
                        slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None),
                    )
                    w_slices = (
                        slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None),
                    )
                    cnt = 0
                    for h in h_slices:
                        for w in w_slices:
                            img_mask[:, h, w, :] = cnt
                            cnt += 1
                    mask_windows = window_partition(img_mask, window_size)
                    mask_windows = mask_windows.view(-1, window_size * window_size)
                    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                    attn_mask = attn_mask.masked_fill(
                        attn_mask != 0, float(-100.0)
                    ).masked_fill(attn_mask == 0, float(0.0))
                else:
                    attn_mask = None
                self.register_buffer(
                    f"attn_mask_layer{i_layer}_block{i_block}",
                    attn_mask,
                    persistent=True,
                )

                # Relative coordinates table
                window_size_tuple = to_2tuple(window_size)
                pretrained_window_size_tuple = to_2tuple(
                    pretrained_window_sizes[i_layer]
                )
                relative_coords_h = torch.arange(
                    -(window_size - 1), window_size, dtype=torch.float32
                )
                relative_coords_w = torch.arange(
                    -(window_size - 1), window_size, dtype=torch.float32
                )
                relative_coords_table = (
                    torch.stack(
                        torch.meshgrid(
                            [relative_coords_h, relative_coords_w], indexing="ij"
                        )
                    )
                    .permute(1, 2, 0)
                    .contiguous()
                    .unsqueeze(0)
                )
                if pretrained_window_size_tuple[0] > 0:
                    relative_coords_table[:, :, :, 0] /= (
                        pretrained_window_size_tuple[0] - 1
                    )
                    relative_coords_table[:, :, :, 1] /= (
                        pretrained_window_size_tuple[1] - 1
                    )
                else:
                    relative_coords_table[:, :, :, 0] /= window_size - 1
                    relative_coords_table[:, :, :, 1] /= window_size - 1
                relative_coords_table *= 8
                relative_coords_table = (
                    torch.sign(relative_coords_table)
                    * torch.log2(torch.abs(relative_coords_table) + 1.0)
                    / np.log2(8)
                )
                self.register_buffer(
                    f"relative_coords_table_layer{i_layer}_block{i_block}",
                    relative_coords_table.float(),
                    persistent=True,
                )

                # Relative position index
                coords_h = torch.arange(window_size)
                coords_w = torch.arange(window_size)
                coords = torch.stack(
                    torch.meshgrid([coords_h, coords_w], indexing="ij")
                )
                coords_flatten = torch.flatten(coords, 1)
                relative_coords = (
                    coords_flatten[:, :, None] - coords_flatten[:, None, :]
                )
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()
                relative_coords[:, :, 0] += window_size - 1
                relative_coords[:, :, 1] += window_size - 1
                relative_coords[:, :, 0] *= 2 * window_size - 1
                relative_position_index = relative_coords.sum(-1)
                self.register_buffer(
                    f"relative_position_index_layer{i_layer}_block{i_block}",
                    relative_position_index.float(),
                    persistent=True,
                )

                # Drop path rate for this block
                self.register_buffer(
                    f"drop_path_rate_layer{i_layer}_block{i_block}",
                    torch.tensor(dpr[dpr_idx], dtype=torch.float),
                )
                dpr_idx += 1

                # Normalization parameters
                self.register_parameter(
                    f"norm1_weight_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.ones(current_dim)),
                )
                self.register_parameter(
                    f"norm1_bias_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(current_dim)),
                )
                self.register_parameter(
                    f"norm2_weight_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.ones(current_dim)),
                )
                self.register_parameter(
                    f"norm2_bias_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(current_dim)),
                )

                # Attention parameters
                self.register_parameter(
                    f"qkv_weight_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(current_dim * 3, current_dim)),
                )
                if qkv_bias:
                    self.register_parameter(
                        f"q_bias_layer{i_layer}_block{i_block}",
                        nn.Parameter(torch.zeros(current_dim)),
                    )
                    self.register_parameter(
                        f"v_bias_layer{i_layer}_block{i_block}",
                        nn.Parameter(torch.zeros(current_dim)),
                    )
                else:
                    self.register_parameter(
                        f"q_bias_layer{i_layer}_block{i_block}", None
                    )
                    self.register_parameter(
                        f"v_bias_layer{i_layer}_block{i_block}", None
                    )
                self.register_parameter(
                    f"proj_weight_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(current_dim, current_dim)),
                )
                self.register_parameter(
                    f"proj_bias_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(current_dim)),
                )
                self.register_parameter(
                    f"logit_scale_layer{i_layer}_block{i_block}",
                    nn.Parameter(
                        torch.log(10 * torch.ones((num_heads[i_layer], 1, 1)))
                    ),
                )
                self.register_parameter(
                    f"cpb_mlp_weight0_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(512, 2)),
                )
                self.register_parameter(
                    f"cpb_mlp_weight1_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(num_heads[i_layer], 512)),
                )
                self.register_parameter(
                    f"cpb_mlp_bias0_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(512)),
                )

                # MLP parameters
                mlp_hidden_dim = int(current_dim * mlp_ratio)
                self.register_parameter(
                    f"fc1_weight_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(mlp_hidden_dim, current_dim)),
                )
                self.register_parameter(
                    f"fc1_bias_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(mlp_hidden_dim)),
                )
                self.register_parameter(
                    f"fc2_weight_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(current_dim, mlp_hidden_dim)),
                )
                self.register_parameter(
                    f"fc2_bias_layer{i_layer}_block{i_block}",
                    nn.Parameter(torch.zeros(current_dim)),
                )

            # Downsample parameters (if not the last layer)
            if i_layer < len(depths) - 1:
                next_dim = int(embed_dim * 2 ** (i_layer + 1))
                self.register_parameter(
                    f"reduction_weight_layer{i_layer}",
                    nn.Parameter(torch.zeros(next_dim, current_dim * 4)),
                )
                self.register_parameter(
                    f"norm_weight_layer{i_layer}", nn.Parameter(torch.ones(next_dim))
                )
                self.register_parameter(
                    f"norm_bias_layer{i_layer}", nn.Parameter(torch.zeros(next_dim))
                )

        ### Final Normalization and Classification Parameters ###
        self.register_parameter("norm_weight", nn.Parameter(torch.ones(num_features)))
        self.register_parameter("norm_bias", nn.Parameter(torch.zeros(num_features)))
        if num_classes > 0:
            self.register_parameter(
                "head_weight", nn.Parameter(torch.zeros(num_classes, num_features))
            )
            self.register_parameter("head_bias", nn.Parameter(torch.zeros(num_classes)))
        else:
            self.register_parameter("head_weight", None)
            self.register_parameter("head_bias", None)

    @staticmethod
    def model_forward(
        x: torch.Tensor, param_dict: dict, buffer_dict: dict
    ) -> torch.Tensor:
        """
        Forward pass of the model using flat parameter and buffer dictionaries.

        Args:
            x (torch.Tensor): Input tensor.
            param_dict (dict): Dictionary of all model parameters.
            buffer_dict (dict): Dictionary of all model buffers.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Patch embedding
        x = patch_embed_forward(x, param_dict, buffer_dict)

        # Position dropout
        x = F.dropout(x, p=float(buffer_dict["drop_rate"]), training=True)

        # Process through layers
        num_layers = int(buffer_dict["num_layers"])
        for i_layer in range(num_layers):
            x = basic_layer_forward(x, param_dict, buffer_dict, i_layer)

        # Final normalization
        x = F.layer_norm(
            x,
            (x.size(-1),),
            weight=param_dict["norm_weight"],
            bias=param_dict["norm_bias"],
        )

        # Global pooling
        x = x.transpose(1, 2)  # B C L
        x = F.adaptive_avg_pool1d(x, 1)  # B C 1
        x = torch.flatten(x, 1)  # B C

        # Classification head
        if "head_weight" in param_dict and param_dict["head_weight"] is not None:
            x = F.linear(x, param_dict["head_weight"], param_dict["head_bias"])

        return x

    def forward(self, x: torch.Tensor, fn=None) -> torch.Tensor:
        # Flatten parameters and buffers
        param_dict = {name: param for name, param in self.named_parameters()}
        buffer_dict = {name: buffer for name, buffer in self.named_buffers()}

        if fn is not None:
            x = fn(x, param_dict, buffer_dict)
        else:
            x = self.model_forward(x, param_dict, buffer_dict)
        return x


batch_size = 10
image_size = 224


def get_inputs():
    return [torch.randn(batch_size, 3, image_size, image_size)]


def get_init_inputs():
    return []


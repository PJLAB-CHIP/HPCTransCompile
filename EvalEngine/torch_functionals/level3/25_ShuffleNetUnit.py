import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, params: nn.ParameterDict, is_training: bool
) -> torch.Tensor:
    """
    Implementation of ShuffleNet unit.

    Args:
        x: Input tensor, shape (batch_size, in_channels, height, width)
        params: Dictionary containing model parameters
        is_training: Whether in training mode

    Returns:
        Output tensor, shape (batch_size, out_channels, height, width)
    """
    # First group conv + bn
    out = F.conv2d(x, params["conv1_weight"], bias=None, groups=params["groups"])
    out = F.batch_norm(
        out,
        params["bn1_running_mean"],
        params["bn1_running_var"],
        params["bn1_weight"],
        params["bn1_bias"],
        training=is_training,
    )
    out = F.relu(out)

    # Depthwise conv + bn
    out = F.conv2d(
        out, params["conv2_weight"], bias=None, padding=1, groups=params["mid_channels"]
    )
    out = F.batch_norm(
        out,
        params["bn2_running_mean"],
        params["bn2_running_var"],
        params["bn2_weight"],
        params["bn2_bias"],
        training=is_training,
    )

    def channel_shuffle(x, groups):
        """
        Functional implementation of channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :param groups: Number of groups for shuffling
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // groups

        # Reshape
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # Transpose
        x = x.transpose(1, 2).contiguous()

        # Flatten
        x = x.view(batch_size, -1, height, width)

        return x

    # Channel shuffle
    out = channel_shuffle(out, params["groups"])

    # Second group conv + bn
    out = F.conv2d(out, params["conv3_weight"], bias=None, groups=params["groups"])
    out = F.batch_norm(
        out,
        params["bn3_running_mean"],
        params["bn3_running_var"],
        params["bn3_weight"],
        params["bn3_bias"],
        training=is_training,
    )
    out = F.relu(out)

    # Shortcut
    if "shortcut_conv_weight" in params:
        shortcut = F.conv2d(x, params["shortcut_conv_weight"], bias=None)
        shortcut = F.batch_norm(
            shortcut,
            params["shortcut_bn_running_mean"],
            params["shortcut_bn_running_var"],
            params["shortcut_bn_weight"],
            params["shortcut_bn_bias"],
            training=is_training,
        )
    else:
        shortcut = x

    out += shortcut
    return out


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param groups: Number of groups for group convolution
        """
        super(Model, self).__init__()

        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.params = nn.ParameterDict()
        self.params["groups"] = groups
        self.params["mid_channels"] = mid_channels

        # First group conv
        conv1 = nn.Conv2d(in_channels, mid_channels, 1, groups=groups, bias=False)
        self.params["conv1_weight"] = nn.Parameter(conv1.weight.data.clone())

        # First bn
        bn1 = nn.BatchNorm2d(mid_channels)
        self.params["bn1_weight"] = nn.Parameter(bn1.weight.data.clone())
        self.params["bn1_bias"] = nn.Parameter(bn1.bias.data.clone())
        self.params["bn1_running_mean"] = nn.Parameter(bn1.running_mean.data.clone())
        self.params["bn1_running_var"] = nn.Parameter(bn1.running_var.data.clone())

        # Depthwise conv
        conv2 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False
        )
        self.params["conv2_weight"] = nn.Parameter(conv2.weight.data.clone())

        # Second bn
        bn2 = nn.BatchNorm2d(mid_channels)
        self.params["bn2_weight"] = nn.Parameter(bn2.weight.data.clone())
        self.params["bn2_bias"] = nn.Parameter(bn2.bias.data.clone())
        self.params["bn2_running_mean"] = nn.Parameter(bn2.running_mean.data.clone())
        self.params["bn2_running_var"] = nn.Parameter(bn2.running_var.data.clone())

        # Second group conv
        conv3 = nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False)
        self.params["conv3_weight"] = nn.Parameter(conv3.weight.data.clone())

        # Third bn
        bn3 = nn.BatchNorm2d(out_channels)
        self.params["bn3_weight"] = nn.Parameter(bn3.weight.data.clone())
        self.params["bn3_bias"] = nn.Parameter(bn3.bias.data.clone())
        self.params["bn3_running_mean"] = nn.Parameter(bn3.running_mean.data.clone())
        self.params["bn3_running_var"] = nn.Parameter(bn3.running_var.data.clone())

        # Shortcut if needed
        if in_channels != out_channels:
            shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            shortcut_bn = nn.BatchNorm2d(out_channels)

            self.params["shortcut_conv_weight"] = nn.Parameter(
                shortcut_conv.weight.data.clone()
            )
            self.params["shortcut_bn_weight"] = nn.Parameter(
                shortcut_bn.weight.data.clone()
            )
            self.params["shortcut_bn_bias"] = nn.Parameter(
                shortcut_bn.bias.data.clone()
            )
            self.params["shortcut_bn_running_mean"] = nn.Parameter(
                shortcut_bn.running_mean.data.clone()
            )
            self.params["shortcut_bn_running_var"] = nn.Parameter(
                shortcut_bn.running_var.data.clone()
            )

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [input_channels, out_channels, groups]

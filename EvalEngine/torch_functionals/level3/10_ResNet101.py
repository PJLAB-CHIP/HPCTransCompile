import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, params: nn.ParameterDict, is_training: bool
) -> torch.Tensor:
    """
    Implements the ResNet101 module.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        params (nn.ParameterDict): Dictionary of parameters
        is_training (bool): Whether to use training mode

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, num_classes)
    """
    # Initial layers
    x = F.conv2d(x, params["conv1_w"].to(x.device), bias=None, stride=2, padding=3)
    x = F.batch_norm(
        x,
        params["bn1_m"].to(x.device),
        params["bn1_v"].to(x.device),
        params["bn1_w"].to(x.device),
        params["bn1_b"].to(x.device),
        training=is_training,
    )
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    def bottleneck_fn(
        x,
        conv1_w,
        conv2_w,
        conv3_w,
        bn1_w,
        bn1_b,
        bn1_m,
        bn1_v,
        bn2_w,
        bn2_b,
        bn2_m,
        bn2_v,
        bn3_w,
        bn3_b,
        bn3_m,
        bn3_v,
        downsample_conv_w=None,
        downsample_bn_w=None,
        downsample_bn_b=None,
        downsample_bn_m=None,
        downsample_bn_v=None,
        stride=1,
        is_training=True,
    ):
        identity = x

        out = F.conv2d(x, conv1_w.to(x.device), bias=None)
        out = F.batch_norm(
            out,
            bn1_m.to(x.device),
            bn1_v.to(x.device),
            bn1_w.to(x.device),
            bn1_b.to(x.device),
            training=is_training,
        )
        out = F.relu(out)

        out = F.conv2d(out, conv2_w.to(x.device), bias=None, stride=stride, padding=1)
        out = F.batch_norm(
            out,
            bn2_m.to(x.device),
            bn2_v.to(x.device),
            bn2_w.to(x.device),
            bn2_b.to(x.device),
            training=is_training,
        )
        out = F.relu(out)

        out = F.conv2d(out, conv3_w.to(x.device), bias=None)
        out = F.batch_norm(
            out,
            bn3_m.to(x.device),
            bn3_v.to(x.device),
            bn3_w.to(x.device),
            bn3_b.to(x.device),
            training=is_training,
        )

        if downsample_conv_w is not None:
            identity = F.conv2d(
                x, downsample_conv_w.to(x.device), bias=None, stride=stride
            )
            identity = F.batch_norm(
                identity,
                downsample_bn_m.to(x.device),
                downsample_bn_v.to(x.device),
                downsample_bn_w.to(x.device),
                downsample_bn_b.to(x.device),
                training=is_training,
            )

        out += identity
        out = F.relu(out)

        return out

    # Layer 1-4
    for layer_idx in range(1, 5):
        blocks = params[f"layer{layer_idx}_blocks"]
        for block_idx in range(len(blocks)):
            block_params = blocks[block_idx]

            downsample_params = None
            if "downsample_conv_w" in block_params:
                downsample_params = [
                    block_params["downsample_conv_w"],
                    block_params["downsample_bn_w"],
                    block_params["downsample_bn_b"],
                    block_params["downsample_bn_m"],
                    block_params["downsample_bn_v"],
                ]

            x = bottleneck_fn(
                x,
                block_params["conv1_w"],
                block_params["conv2_w"],
                block_params["conv3_w"],
                block_params["bn1_w"],
                block_params["bn1_b"],
                block_params["bn1_m"],
                block_params["bn1_v"],
                block_params["bn2_w"],
                block_params["bn2_b"],
                block_params["bn2_m"],
                block_params["bn2_v"],
                block_params["bn3_w"],
                block_params["bn3_b"],
                block_params["bn3_m"],
                block_params["bn3_v"],
                *(downsample_params if downsample_params else [None] * 5),
                stride=2 if block_idx == 0 and layer_idx > 1 else 1,
                is_training=is_training,
            )

    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = F.linear(x, params["fc_w"].to(x.device), params["fc_b"].to(x.device))

    return x


class Model(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(Model, self).__init__()
        self.params = nn.ParameterDict()
        in_channels = 64
        expansion = 4

        # Initial layers
        conv1 = nn.Conv2d(
            3, in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        bn1 = nn.BatchNorm2d(in_channels)
        self.params["conv1_w"] = nn.Parameter(conv1.weight.data.clone())
        self.params["bn1_w"] = nn.Parameter(bn1.weight.data.clone())
        self.params["bn1_b"] = nn.Parameter(bn1.bias.data.clone())
        self.params["bn1_m"] = nn.Parameter(bn1.running_mean.data.clone())
        self.params["bn1_v"] = nn.Parameter(bn1.running_var.data.clone())

        # Layers 1-4
        channels = [64, 128, 256, 512]
        for layer_idx, (out_channels, num_blocks) in enumerate(
            zip(channels, layers), 1
        ):
            layer_blocks = []

            for block_idx in range(num_blocks):
                block_in_channels = (
                    in_channels if block_idx == 0 else out_channels * expansion
                )

                # Create block parameters
                block_params = {}

                # First block may have downsample
                if block_idx == 0 and (
                    layer_idx > 1 or block_in_channels != out_channels * expansion
                ):
                    downsample_conv = nn.Conv2d(
                        block_in_channels,
                        out_channels * expansion,
                        kernel_size=1,
                        stride=2 if layer_idx > 1 else 1,
                        bias=False,
                    )
                    downsample_bn = nn.BatchNorm2d(out_channels * expansion)

                    block_params["downsample_conv_w"] = nn.Parameter(
                        downsample_conv.weight.data.clone()
                    )
                    block_params["downsample_bn_w"] = nn.Parameter(
                        downsample_bn.weight.data.clone()
                    )
                    block_params["downsample_bn_b"] = nn.Parameter(
                        downsample_bn.bias.data.clone()
                    )
                    block_params["downsample_bn_m"] = nn.Parameter(
                        downsample_bn.running_mean.data.clone()
                    )
                    block_params["downsample_bn_v"] = nn.Parameter(
                        downsample_bn.running_var.data.clone()
                    )

                conv1 = nn.Conv2d(
                    block_in_channels, out_channels, kernel_size=1, bias=False
                )
                bn1 = nn.BatchNorm2d(out_channels)
                conv2 = nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2 if block_idx == 0 and layer_idx > 1 else 1,
                    padding=1,
                    bias=False,
                )
                bn2 = nn.BatchNorm2d(out_channels)
                conv3 = nn.Conv2d(
                    out_channels, out_channels * expansion, kernel_size=1, bias=False
                )
                bn3 = nn.BatchNorm2d(out_channels * expansion)

                block_params["conv1_w"] = nn.Parameter(conv1.weight.data.clone())
                block_params["bn1_w"] = nn.Parameter(bn1.weight.data.clone())
                block_params["bn1_b"] = nn.Parameter(bn1.bias.data.clone())
                block_params["bn1_m"] = nn.Parameter(bn1.running_mean.data.clone())
                block_params["bn1_v"] = nn.Parameter(bn1.running_var.data.clone())

                block_params["conv2_w"] = nn.Parameter(conv2.weight.data.clone())
                block_params["bn2_w"] = nn.Parameter(bn2.weight.data.clone())
                block_params["bn2_b"] = nn.Parameter(bn2.bias.data.clone())
                block_params["bn2_m"] = nn.Parameter(bn2.running_mean.data.clone())
                block_params["bn2_v"] = nn.Parameter(bn2.running_var.data.clone())

                block_params["conv3_w"] = nn.Parameter(conv3.weight.data.clone())
                block_params["bn3_w"] = nn.Parameter(bn3.weight.data.clone())
                block_params["bn3_b"] = nn.Parameter(bn3.bias.data.clone())
                block_params["bn3_m"] = nn.Parameter(bn3.running_mean.data.clone())
                block_params["bn3_v"] = nn.Parameter(bn3.running_var.data.clone())

                layer_blocks.append(block_params)

            self.params[f"layer{layer_idx}_blocks"] = layer_blocks
            in_channels = out_channels * expansion

        # Final FC layer
        fc = nn.Linear(512 * expansion, num_classes)
        self.params["fc_w"] = nn.Parameter(fc.weight.data.clone())
        self.params["fc_b"] = nn.Parameter(fc.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


# Test configurations
batch_size = 10
height = 224
width = 224
layers = [3, 4, 23, 3]
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]


def get_init_inputs():
    return [layers, num_classes]

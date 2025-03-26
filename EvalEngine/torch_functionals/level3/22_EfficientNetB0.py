import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, params: nn.ParameterDict, is_training: bool
) -> torch.Tensor:
    """
    Implementation of EfficientNetB0.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).
        params (nn.ParameterDict): Parameter dictionary containing the model parameters.
        is_training (bool): Whether the model is in training mode.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1000).
    """
    # Initial conv
    x = F.conv2d(x, params["conv1_weight"], None, stride=2, padding=1)
    x = F.batch_norm(
        x,
        params["bn1_running_mean"],
        params["bn1_running_var"],
        params["bn1_weight"],
        params["bn1_bias"],
        training=is_training,
    )
    x = F.relu(x)

    # MBConv blocks
    block_configs = [
        (1, 1),
        (6, 2),
        (6, 1),
        (6, 2),
        (6, 1),
        (6, 2),
        (6, 1),
        (6, 1),
        (6, 1),
        (6, 2),
        (6, 1),
        (6, 1),
        (6, 1),
    ]

    def mbconv_fn(x, params, expand_ratio, stride, use_residual, is_training):
        """
        Functional implementation of MBConv block.
        """
        identity = x
        hidden_dim = x.size(1) * expand_ratio

        if expand_ratio != 1:
            # Expand conv
            x = F.conv2d(x, params["expand_conv_weight"], None)
            x = F.batch_norm(
                x,
                params["expand_conv_bn_running_mean"],
                params["expand_conv_bn_running_var"],
                params["expand_conv_bn_weight"],
                params["expand_conv_bn_bias"],
                training=is_training,
            )
            x = F.relu6(x)

        # Depthwise conv
        x = F.conv2d(
            x,
            params["depthwise_conv_weight"],
            None,
            stride=stride,
            padding=(params["depthwise_conv_weight"].size(2) - 1) // 2,
            groups=hidden_dim,
        )
        x = F.batch_norm(
            x,
            params["depthwise_conv_bn_running_mean"],
            params["depthwise_conv_bn_running_var"],
            params["depthwise_conv_bn_weight"],
            params["depthwise_conv_bn_bias"],
            training=is_training,
        )
        x = F.relu6(x)

        # Project conv
        x = F.conv2d(x, params["project_conv_weight"], None)
        x = F.batch_norm(
            x,
            params["project_conv_bn_running_mean"],
            params["project_conv_bn_running_var"],
            params["project_conv_bn_weight"],
            params["project_conv_bn_bias"],
            training=is_training,
        )

        if use_residual:
            x += identity

        return x

    for i, (expand_ratio, stride) in enumerate(block_configs):
        x = mbconv_fn(
            x,
            params[f"block{i}"],
            expand_ratio,
            stride,
            stride == 1
            and x.size(1) == params[f"block{i}"]["project_conv_weight"].size(0),
            is_training,
        )

    # Final conv
    x = F.conv2d(x, params["conv2_weight"], None)
    x = F.batch_norm(
        x,
        params["bn2_running_mean"],
        params["bn2_running_var"],
        params["bn2_weight"],
        params["bn2_bias"],
        training=is_training,
    )
    x = F.relu(x)

    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    x = F.linear(x, params["fc_weight"], params["fc_bias"])

    return x


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        self.params = nn.ParameterDict()

        # Initial conv params
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(32)
        self.params["conv1_weight"] = nn.Parameter(conv1.weight.data.clone())
        self.params["bn1_weight"] = nn.Parameter(bn1.weight.data.clone())
        self.params["bn1_bias"] = nn.Parameter(bn1.bias.data.clone())
        self.params["bn1_running_mean"] = nn.Parameter(bn1.running_mean.data.clone())
        self.params["bn1_running_var"] = nn.Parameter(bn1.running_var.data.clone())

        # MBConv blocks params
        block_configs = [
            (32, 16, 3, 1, 1),
            (16, 24, 3, 2, 6),
            (24, 24, 3, 1, 6),
            (24, 40, 5, 2, 6),
            (40, 40, 5, 1, 6),
            (40, 80, 3, 2, 6),
            (80, 80, 3, 1, 6),
            (80, 112, 5, 1, 6),
            (112, 112, 5, 1, 6),
            (112, 192, 5, 2, 6),
            (192, 192, 5, 1, 6),
            (192, 192, 5, 1, 6),
            (192, 320, 3, 1, 6),
        ]

        for i, (in_c, out_c, k, s, e) in enumerate(block_configs):
            block = MBConv(in_c, out_c, k, s, e)
            block_params = nn.ParameterDict()

            if e != 1:
                block_params["expand_conv_weight"] = nn.Parameter(
                    block.expand_conv[0].weight.data.clone()
                )
                block_params["expand_conv_bn_weight"] = nn.Parameter(
                    block.expand_conv[1].weight.data.clone()
                )
                block_params["expand_conv_bn_bias"] = nn.Parameter(
                    block.expand_conv[1].bias.data.clone()
                )
                block_params["expand_conv_bn_running_mean"] = nn.Parameter(
                    block.expand_conv[1].running_mean.data.clone()
                )
                block_params["expand_conv_bn_running_var"] = nn.Parameter(
                    block.expand_conv[1].running_var.data.clone()
                )

            block_params["depthwise_conv_weight"] = nn.Parameter(
                block.depthwise_conv[0].weight.data.clone()
            )
            block_params["depthwise_conv_bn_weight"] = nn.Parameter(
                block.depthwise_conv[1].weight.data.clone()
            )
            block_params["depthwise_conv_bn_bias"] = nn.Parameter(
                block.depthwise_conv[1].bias.data.clone()
            )
            block_params["depthwise_conv_bn_running_mean"] = nn.Parameter(
                block.depthwise_conv[1].running_mean.data.clone()
            )
            block_params["depthwise_conv_bn_running_var"] = nn.Parameter(
                block.depthwise_conv[1].running_var.data.clone()
            )

            block_params["project_conv_weight"] = nn.Parameter(
                block.project_conv[0].weight.data.clone()
            )
            block_params["project_conv_bn_weight"] = nn.Parameter(
                block.project_conv[1].weight.data.clone()
            )
            block_params["project_conv_bn_bias"] = nn.Parameter(
                block.project_conv[1].bias.data.clone()
            )
            block_params["project_conv_bn_running_mean"] = nn.Parameter(
                block.project_conv[1].running_mean.data.clone()
            )
            block_params["project_conv_bn_running_var"] = nn.Parameter(
                block.project_conv[1].running_var.data.clone()
            )

            self.params[f"block{i}"] = block_params

        # Final conv params
        conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        bn2 = nn.BatchNorm2d(1280)
        self.params["conv2_weight"] = nn.Parameter(conv2.weight.data.clone())
        self.params["bn2_weight"] = nn.Parameter(bn2.weight.data.clone())
        self.params["bn2_bias"] = nn.Parameter(bn2.bias.data.clone())
        self.params["bn2_running_mean"] = nn.Parameter(bn2.running_mean.data.clone())
        self.params["bn2_running_var"] = nn.Parameter(bn2.running_var.data.clone())

        # FC params
        fc = nn.Linear(1280, num_classes)
        self.params["fc_weight"] = nn.Parameter(fc.weight.data.clone())
        self.params["fc_bias"] = nn.Parameter(fc.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(MBConv, self).__init__()

        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )


batch_size = 10
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]


def get_init_inputs():
    return [num_classes]

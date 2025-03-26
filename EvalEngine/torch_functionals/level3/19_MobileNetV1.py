import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, params: nn.ParameterDict, is_training: bool
) -> torch.Tensor:
    """
    Implements the MobileNetV1 forward pass.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        params (nn.ParameterDict): Dictionary of parameters
        is_training (bool): Whether to use training mode

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, num_classes)
    """

    def conv_bn_fn(
        x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var, stride, is_training
    ):
        x = F.conv2d(x, conv_weight, None, (stride, stride), (1, 1))
        x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=is_training)
        x = F.relu(x)
        return x

    def conv_dw_fn(
        x,
        dw_conv_weight,
        dw_bn_weight,
        dw_bn_bias,
        dw_bn_mean,
        dw_bn_var,
        pw_conv_weight,
        pw_bn_weight,
        pw_bn_bias,
        pw_bn_mean,
        pw_bn_var,
        stride,
        is_training,
    ):
        # Depthwise
        x = F.conv2d(
            x,
            dw_conv_weight,
            None,
            (stride, stride),
            (1, 1),
            groups=dw_conv_weight.size(0),
        )
        x = F.batch_norm(
            x, dw_bn_mean, dw_bn_var, dw_bn_weight, dw_bn_bias, training=is_training
        )
        x = F.relu(x)

        # Pointwise
        x = F.conv2d(x, pw_conv_weight, None, (1, 1), (0, 0))
        x = F.batch_norm(
            x, pw_bn_mean, pw_bn_var, pw_bn_weight, pw_bn_bias, training=is_training
        )
        x = F.relu(x)
        return x

    # First conv+bn+relu
    x = conv_bn_fn(
        x,
        params["conv0_weight"],
        params["bn0_weight"],
        params["bn0_bias"],
        params["bn0_mean"],
        params["bn0_var"],
        2,
        is_training,
    )

    # 13 conv_dw blocks
    for i in range(13):
        x = conv_dw_fn(
            x,
            params[f"conv{i+1}_dw_weight"],
            params[f"bn{i+1}_dw_weight"],
            params[f"bn{i+1}_dw_bias"],
            params[f"bn{i+1}_dw_mean"],
            params[f"bn{i+1}_dw_var"],
            params[f"conv{i+1}_pw_weight"],
            params[f"bn{i+1}_pw_weight"],
            params[f"bn{i+1}_pw_bias"],
            params[f"bn{i+1}_pw_mean"],
            params[f"bn{i+1}_pw_var"],
            2 if i in [1, 3, 5, 11] else 1,
            is_training,
        )

    # Average pooling
    x = F.avg_pool2d(x, 7)

    # Flatten and FC
    x = x.view(x.size(0), -1)
    x = F.linear(x, params["fc_weight"], params["fc_bias"])

    return x


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(Model, self).__init__()

        def conv_bn(inp, oup, stride):
            conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
            bn = nn.BatchNorm2d(oup)
            return nn.Sequential(conv, bn, nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            conv_dw = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
            bn_dw = nn.BatchNorm2d(inp)
            conv_pw = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
            bn_pw = nn.BatchNorm2d(oup)
            return nn.Sequential(
                conv_dw,
                bn_dw,
                nn.ReLU(inplace=True),
                conv_pw,
                bn_pw,
                nn.ReLU(inplace=True),
            )

        self.params = nn.ParameterDict()

        # Build model and extract parameters
        model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )

        # Extract first conv+bn parameters
        self.params["conv0_weight"] = nn.Parameter(model[0][0].weight.data.clone())
        self.params["bn0_weight"] = nn.Parameter(model[0][1].weight.data.clone())
        self.params["bn0_bias"] = nn.Parameter(model[0][1].bias.data.clone())
        self.params["bn0_mean"] = nn.Parameter(model[0][1].running_mean.data.clone())
        self.params["bn0_var"] = nn.Parameter(model[0][1].running_var.data.clone())

        # Extract parameters from conv_dw blocks
        for i in range(13):
            layer = model[i + 1]
            # Depthwise conv+bn
            self.params[f"conv{i+1}_dw_weight"] = nn.Parameter(
                layer[0].weight.data.clone()
            )
            self.params[f"bn{i+1}_dw_weight"] = nn.Parameter(
                layer[1].weight.data.clone()
            )
            self.params[f"bn{i+1}_dw_bias"] = nn.Parameter(layer[1].bias.data.clone())
            self.params[f"bn{i+1}_dw_mean"] = nn.Parameter(
                layer[1].running_mean.data.clone()
            )
            self.params[f"bn{i+1}_dw_var"] = nn.Parameter(
                layer[1].running_var.data.clone()
            )

            # Pointwise conv+bn
            self.params[f"conv{i+1}_pw_weight"] = nn.Parameter(
                layer[3].weight.data.clone()
            )
            self.params[f"bn{i+1}_pw_weight"] = nn.Parameter(
                layer[4].weight.data.clone()
            )
            self.params[f"bn{i+1}_pw_bias"] = nn.Parameter(layer[4].bias.data.clone())
            self.params[f"bn{i+1}_pw_mean"] = nn.Parameter(
                layer[4].running_mean.data.clone()
            )
            self.params[f"bn{i+1}_pw_var"] = nn.Parameter(
                layer[4].running_var.data.clone()
            )

        # FC layer parameters
        fc = nn.Linear(int(1024 * alpha), num_classes)
        fc_weight = nn.Parameter(fc.weight.data.clone())
        fc_bias = nn.Parameter(fc.bias.data.clone())
        self.params["fc_weight"] = fc_weight
        self.params["fc_bias"] = fc_bias

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000
alpha = 1.0


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [num_classes, input_channels, alpha]

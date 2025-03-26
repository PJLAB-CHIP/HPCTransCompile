import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, params: nn.ParameterDict, is_training: bool
) -> torch.Tensor:
    """
    Implementation of EfficientNetB1.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, 3, 240, 240).
        params (nn.ParameterDict): Parameter dictionary containing the model parameters.
        is_training (bool): Whether the model is in training mode.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1000).
    """
    # Initial conv
    x = F.conv2d(x, params["conv1_w"], bias=None, stride=2, padding=1)
    x = F.batch_norm(
        x,
        params["bn1_rm"],
        params["bn1_rv"],
        params["bn1_w"],
        params["bn1_b"],
        training=is_training,
    )
    x = F.relu(x)

    # MBConv blocks configs - strides for each block
    strides = [1, 2, 2, 2, 1, 2, 1]

    def mbconv_block_fn(
        x,
        conv1_w,
        conv1_bn_w,
        conv1_bn_b,
        conv1_bn_rm,
        conv1_bn_rv,
        conv2_w,
        conv2_bn_w,
        conv2_bn_b,
        conv2_bn_rm,
        conv2_bn_rv,
        conv3_w,
        conv3_bn_w,
        conv3_bn_b,
        conv3_bn_rm,
        conv3_bn_rv,
        stride,
        is_training,
    ):
        """
        Functional implementation of MBConv block
        """
        # Expansion conv 1x1
        x = F.conv2d(x, conv1_w, bias=None, stride=1, padding=0)
        x = F.batch_norm(
            x, conv1_bn_rm, conv1_bn_rv, conv1_bn_w, conv1_bn_b, training=is_training
        )
        x = F.relu6(x)

        # Depthwise conv 3x3
        x = F.conv2d(
            x, conv2_w, bias=None, stride=stride, padding=1, groups=conv2_w.shape[0]
        )
        x = F.batch_norm(
            x, conv2_bn_rm, conv2_bn_rv, conv2_bn_w, conv2_bn_b, training=is_training
        )
        x = F.relu6(x)

        # Projection conv 1x1
        x = F.conv2d(x, conv3_w, bias=None, stride=1, padding=0)
        x = F.batch_norm(
            x, conv3_bn_rm, conv3_bn_rv, conv3_bn_w, conv3_bn_b, training=is_training
        )

        return x

    # MBConv blocks
    for i, stride in enumerate(strides, 1):
        prefix = f"mbconv{i}_"
        x = mbconv_block_fn(
            x,
            params[prefix + "conv1_w"],
            params[prefix + "conv1_bn_w"],
            params[prefix + "conv1_bn_b"],
            params[prefix + "conv1_bn_rm"],
            params[prefix + "conv1_bn_rv"],
            params[prefix + "conv2_w"],
            params[prefix + "conv2_bn_w"],
            params[prefix + "conv2_bn_b"],
            params[prefix + "conv2_bn_rm"],
            params[prefix + "conv2_bn_rv"],
            params[prefix + "conv3_w"],
            params[prefix + "conv3_bn_w"],
            params[prefix + "conv3_bn_b"],
            params[prefix + "conv3_bn_rm"],
            params[prefix + "conv3_bn_rv"],
            stride,
            is_training,
        )

    # Final layers
    x = F.conv2d(x, params["conv2_w"], bias=None, stride=1, padding=0)
    x = F.batch_norm(
        x,
        params["bn2_rm"],
        params["bn2_rv"],
        params["bn2_w"],
        params["bn2_b"],
        training=is_training,
    )
    x = F.relu(x)

    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = F.linear(x, params["fc_w"], params["fc_b"])

    return x


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        self.params = nn.ParameterDict()

        # Initial conv
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(32)
        self.params["conv1_w"] = nn.Parameter(conv1.weight.data.clone())
        self.params["bn1_w"] = nn.Parameter(bn1.weight.data.clone())
        self.params["bn1_b"] = nn.Parameter(bn1.bias.data.clone())
        self.params["bn1_rm"] = nn.Parameter(bn1.running_mean.data.clone())
        self.params["bn1_rv"] = nn.Parameter(bn1.running_var.data.clone())

        # MBConv blocks configs
        configs = [
            (32, 16, 1, 1),
            (16, 24, 2, 6),
            (24, 40, 2, 6),
            (40, 80, 2, 6),
            (80, 112, 1, 6),
            (112, 192, 2, 6),
            (192, 320, 1, 6),
        ]

        # Extract parameters from each MBConv block
        for i, (in_c, out_c, stride, expand) in enumerate(configs, 1):
            hidden_dim = round(in_c * expand)
            prefix = f"mbconv{i}_"

            # Expansion conv
            conv1 = nn.Conv2d(in_c, hidden_dim, 1, 1, 0, bias=False)
            bn1 = nn.BatchNorm2d(hidden_dim)
            self.params[prefix + "conv1_w"] = nn.Parameter(conv1.weight.data.clone())
            self.params[prefix + "conv1_bn_w"] = nn.Parameter(bn1.weight.data.clone())
            self.params[prefix + "conv1_bn_b"] = nn.Parameter(bn1.bias.data.clone())
            self.params[prefix + "conv1_bn_rm"] = nn.Parameter(
                bn1.running_mean.data.clone()
            )
            self.params[prefix + "conv1_bn_rv"] = nn.Parameter(
                bn1.running_var.data.clone()
            )

            # Depthwise conv
            conv2 = nn.Conv2d(
                hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
            )
            bn2 = nn.BatchNorm2d(hidden_dim)
            self.params[prefix + "conv2_w"] = nn.Parameter(conv2.weight.data.clone())
            self.params[prefix + "conv2_bn_w"] = nn.Parameter(bn2.weight.data.clone())
            self.params[prefix + "conv2_bn_b"] = nn.Parameter(bn2.bias.data.clone())
            self.params[prefix + "conv2_bn_rm"] = nn.Parameter(
                bn2.running_mean.data.clone()
            )
            self.params[prefix + "conv2_bn_rv"] = nn.Parameter(
                bn2.running_var.data.clone()
            )

            # Projection conv
            conv3 = nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False)
            bn3 = nn.BatchNorm2d(out_c)
            self.params[prefix + "conv3_w"] = nn.Parameter(conv3.weight.data.clone())
            self.params[prefix + "conv3_bn_w"] = nn.Parameter(bn3.weight.data.clone())
            self.params[prefix + "conv3_bn_b"] = nn.Parameter(bn3.bias.data.clone())
            self.params[prefix + "conv3_bn_rm"] = nn.Parameter(
                bn3.running_mean.data.clone()
            )
            self.params[prefix + "conv3_bn_rv"] = nn.Parameter(
                bn3.running_var.data.clone()
            )

        # Final conv
        conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        bn2 = nn.BatchNorm2d(1280)
        self.params["conv2_w"] = nn.Parameter(conv2.weight.data.clone())
        self.params["bn2_w"] = nn.Parameter(bn2.weight.data.clone())
        self.params["bn2_b"] = nn.Parameter(bn2.bias.data.clone())
        self.params["bn2_rm"] = nn.Parameter(bn2.running_mean.data.clone())
        self.params["bn2_rv"] = nn.Parameter(bn2.running_var.data.clone())

        # FC layer
        fc = nn.Linear(1280, num_classes)
        self.params["fc_w"] = nn.Parameter(fc.weight.data.clone())
        self.params["fc_b"] = nn.Parameter(fc.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


# Test code
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, *input_shape)]


def get_init_inputs():
    return [num_classes]

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, params: nn.ParameterDict, is_training: bool):
    """
    Implements the ResNet18 module.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        params (nn.ParameterDict): Dictionary of parameters
        is_training (bool): Whether to use training mode

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, num_classes)
    """
    # Initial layers
    x = F.conv2d(x, params["conv1_weight"], None, stride=2, padding=3)
    x = F.batch_norm(
        x,
        params["bn1_running_mean"],
        params["bn1_running_var"],
        params["bn1_weight"],
        params["bn1_bias"],
        is_training,
    )
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    def basic_block_fn(
        x,
        conv1_w,
        conv1_b,
        bn1_w,
        bn1_b,
        bn1_mean,
        bn1_var,
        conv2_w,
        conv2_b,
        bn2_w,
        bn2_b,
        bn2_mean,
        bn2_var,
        downsample_conv_w=None,
        downsample_conv_b=None,
        downsample_bn_w=None,
        downsample_bn_b=None,
        downsample_bn_mean=None,
        downsample_bn_var=None,
        stride=1,
        is_training=True,
    ):
        identity = x

        out = F.conv2d(x, conv1_w, conv1_b, stride=stride, padding=1)
        out = F.batch_norm(out, bn1_mean, bn1_var, bn1_w, bn1_b, is_training)
        out = F.relu(out)

        out = F.conv2d(out, conv2_w, conv2_b, stride=1, padding=1)
        out = F.batch_norm(out, bn2_mean, bn2_var, bn2_w, bn2_b, is_training)

        if downsample_conv_w is not None:
            identity = F.conv2d(x, downsample_conv_w, downsample_conv_b, stride=stride)
            identity = F.batch_norm(
                identity,
                downsample_bn_mean,
                downsample_bn_var,
                downsample_bn_w,
                downsample_bn_b,
                is_training,
            )

        out += identity
        out = F.relu(out)
        return out

    # Layer blocks
    for i in range(1, 5):
        layer_name = f"layer{i}"
        for j in range(2):
            block_name = f"{layer_name}_{j}"
            stride = 2 if i > 1 and j == 0 else 1

            # Basic block parameters
            conv1_w = params[f"{block_name}_conv1_weight"]
            bn1_w = params[f"{block_name}_bn1_weight"]
            bn1_b = params[f"{block_name}_bn1_bias"]
            bn1_mean = params[f"{block_name}_bn1_running_mean"]
            bn1_var = params[f"{block_name}_bn1_running_var"]

            conv2_w = params[f"{block_name}_conv2_weight"]
            bn2_w = params[f"{block_name}_bn2_weight"]
            bn2_b = params[f"{block_name}_bn2_bias"]
            bn2_mean = params[f"{block_name}_bn2_running_mean"]
            bn2_var = params[f"{block_name}_bn2_running_var"]

            # Downsample parameters if they exist
            has_downsample = f"{block_name}_downsample_0_weight" in params
            downsample_args = {}
            if has_downsample:
                downsample_args = {
                    "downsample_conv_w": params[f"{block_name}_downsample_0_weight"],
                    "downsample_bn_w": params[f"{block_name}_downsample_1_weight"],
                    "downsample_bn_b": params[f"{block_name}_downsample_1_bias"],
                    "downsample_bn_mean": params[
                        f"{block_name}_downsample_1_running_mean"
                    ],
                    "downsample_bn_var": params[
                        f"{block_name}_downsample_1_running_var"
                    ],
                }

            x = basic_block_fn(
                x,
                conv1_w,
                None,
                bn1_w,
                bn1_b,
                bn1_mean,
                bn1_var,
                conv2_w,
                None,
                bn2_w,
                bn2_b,
                bn2_mean,
                bn2_var,
                stride=stride,
                is_training=is_training,
                **downsample_args,
            )

    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = F.linear(x, params["fc_weight"], params["fc_bias"])
    return x


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        self.params = nn.ParameterDict()
        model = OriginalModel(num_classes)  # Create temporary model to copy parameters

        # Copy all parameters
        for name, param in model.named_parameters():
            self.params[name.replace(".", "_")] = nn.Parameter(param.data.clone())

        # Copy all buffers (running means and vars) and add them to params
        for name, buf in model.named_buffers():
            # Register buffer as usual
            self.register_buffer(name.replace(".", "_"), buf.data.clone())
            # Add to params dictionary as a float tensor without requiring gradients
            self.params[name.replace(".", "_")] = buf.data.clone().detach().float()

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class OriginalModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(OriginalModel, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


# Test code
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)


def get_inputs():
    return [torch.randn(input_shape)]


def get_init_inputs():
    return [num_classes]

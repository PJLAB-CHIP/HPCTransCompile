import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, params: nn.ParameterDict, is_training: bool
) -> torch.Tensor:
    """
    Implementation of EfficientNetB2

    Args:
        x: Input tensor of shape (batch_size, 3, 224, 224).
        params: A nn.ParameterDict containing model parameters.
        is_training: Whether the model is in training mode.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1000).
    """
    # Initial conv
    x = F.conv2d(x, params["conv1_weight"], None, stride=2, padding=1)
    x = F.batch_norm(
        x,
        params["bn1_mean"],
        params["bn1_var"],
        params["bn1_weight"],
        params["bn1_bias"],
        is_training,
    )
    x = F.relu(x, inplace=True)

    def mbconv_block_fn(x, params, stride, expand_ratio, is_training):
        """
        Functional implementation of MBConv block
        """
        in_channels = x.size(1)
        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            x = F.conv2d(x, params["expand_conv_weight"], None)
            x = F.batch_norm(
                x,
                params["expand_bn_mean"],
                params["expand_bn_var"],
                params["expand_bn_weight"],
                params["expand_bn_bias"],
                is_training,
            )
            x = F.relu(x, inplace=True)
        else:
            expanded_channels = in_channels

        # Depthwise conv
        x = F.conv2d(
            x,
            params["dw_conv_weight"],
            None,
            stride=stride,
            padding=1,
            groups=expanded_channels,
        )
        x = F.batch_norm(
            x,
            params["dw_bn_mean"],
            params["dw_bn_var"],
            params["dw_bn_weight"],
            params["dw_bn_bias"],
            is_training,
        )
        x = F.relu(x, inplace=True)

        # Squeeze and Excitation
        se = F.adaptive_avg_pool2d(x, (1, 1))
        se = F.conv2d(se, params["se_reduce_weight"], None)
        se = F.relu(se, inplace=True)
        se = F.conv2d(se, params["se_expand_weight"], None)
        se = torch.sigmoid(se)
        x = se
        # x = x * se

        # Output phase
        x = F.conv2d(x, params["project_conv_weight"], None)
        x = F.batch_norm(
            x,
            params["project_bn_mean"],
            params["project_bn_var"],
            params["project_bn_weight"],
            params["project_bn_bias"],
            is_training,
        )

        return x

    # MBConv blocks
    mbconv_configs = [(1, 3), (2, 6), (2, 6), (2, 6), (1, 6)]
    for i, (stride, expand_ratio) in enumerate(mbconv_configs, 1):
        block_params = {
            k.replace(f"mbconv{i}_", ""): v
            for k, v in params.items()
            if k.startswith(f"mbconv{i}_")
        }
        x = mbconv_block_fn(x, block_params, stride, expand_ratio, is_training)

    # Final layers
    x = F.conv2d(x, params["conv_final_weight"], None)
    x = F.batch_norm(
        x,
        params["bn_final_mean"],
        params["bn_final_var"],
        params["bn_final_weight"],
        params["bn_final_bias"],
        is_training,
    )
    x = F.relu(x, inplace=True)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = F.linear(x, params["fc_weight"], params["fc_bias"])

    return x


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        # Create the original model to ensure identical initialization
        original_model = nn.Module()
        original_model.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        original_model.bn1 = nn.BatchNorm2d(32)
        original_model.relu = nn.ReLU(inplace=True)

        # MBConv blocks
        configs = [
            (32, 96, 1, 3),
            (96, 144, 2, 6),
            (144, 192, 2, 6),
            (192, 288, 2, 6),
            (288, 384, 1, 6),
        ]

        for i, (in_c, out_c, stride, expand) in enumerate(configs, 1):
            expanded_c = in_c * expand
            block = nn.Sequential()

            if expand != 1:
                block.add_module(
                    "expand_conv", nn.Conv2d(in_c, expanded_c, 1, bias=False)
                )
                block.add_module("expand_bn", nn.BatchNorm2d(expanded_c))
                block.add_module("expand_relu", nn.ReLU(inplace=True))

            block.add_module(
                "dw_conv",
                nn.Conv2d(
                    expanded_c,
                    expanded_c,
                    3,
                    stride=stride,
                    padding=1,
                    groups=expanded_c,
                    bias=False,
                ),
            )
            block.add_module("dw_bn", nn.BatchNorm2d(expanded_c))
            block.add_module("dw_relu", nn.ReLU(inplace=True))

            block.add_module("se_pool", nn.AdaptiveAvgPool2d((1, 1)))
            block.add_module(
                "se_reduce", nn.Conv2d(expanded_c, expanded_c // 4, 1, bias=False)
            )
            block.add_module("se_reduce_relu", nn.ReLU(inplace=True))
            block.add_module(
                "se_expand", nn.Conv2d(expanded_c // 4, expanded_c, 1, bias=False)
            )
            block.add_module("se_sigmoid", nn.Sigmoid())

            block.add_module(
                "project_conv", nn.Conv2d(expanded_c, out_c, 1, bias=False)
            )
            block.add_module("project_bn", nn.BatchNorm2d(out_c))

            setattr(original_model, f"mbconv{i}", block)

        original_model.conv_final = nn.Conv2d(384, 1408, 1, bias=False)
        original_model.bn_final = nn.BatchNorm2d(1408)
        original_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        original_model.fc = nn.Linear(1408, num_classes)

        # Initialize parameters and buffers
        self.params = nn.ParameterDict()

        # Copy initial conv parameters
        self.params["conv1_weight"] = nn.Parameter(original_model.conv1.weight.data)
        self.params["bn1_weight"] = nn.Parameter(original_model.bn1.weight.data)
        self.params["bn1_bias"] = nn.Parameter(original_model.bn1.bias.data)
        self.register_buffer("bn1_mean", original_model.bn1.running_mean)
        self.register_buffer("bn1_var", original_model.bn1.running_var)

        # Copy MBConv block parameters
        for i in range(1, 6):
            block = getattr(original_model, f"mbconv{i}")
            prefix = f"mbconv{i}_"

            if hasattr(block, "expand_conv"):
                self.params[prefix + "expand_conv_weight"] = nn.Parameter(
                    block.expand_conv.weight.data
                )
                self.params[prefix + "expand_bn_weight"] = nn.Parameter(
                    block.expand_bn.weight.data
                )
                self.params[prefix + "expand_bn_bias"] = nn.Parameter(
                    block.expand_bn.bias.data
                )
                self.register_buffer(
                    prefix + "expand_bn_mean", block.expand_bn.running_mean
                )
                self.register_buffer(
                    prefix + "expand_bn_var", block.expand_bn.running_var
                )

            self.params[prefix + "dw_conv_weight"] = nn.Parameter(
                block.dw_conv.weight.data
            )
            self.params[prefix + "dw_bn_weight"] = nn.Parameter(block.dw_bn.weight.data)
            self.params[prefix + "dw_bn_bias"] = nn.Parameter(block.dw_bn.bias.data)
            self.register_buffer(prefix + "dw_bn_mean", block.dw_bn.running_mean)
            self.register_buffer(prefix + "dw_bn_var", block.dw_bn.running_var)

            self.params[prefix + "se_reduce_weight"] = nn.Parameter(
                block.se_reduce.weight.data
            )
            self.params[prefix + "se_expand_weight"] = nn.Parameter(
                block.se_expand.weight.data
            )

            self.params[prefix + "project_conv_weight"] = nn.Parameter(
                block.project_conv.weight.data
            )
            self.params[prefix + "project_bn_weight"] = nn.Parameter(
                block.project_bn.weight.data
            )
            self.params[prefix + "project_bn_bias"] = nn.Parameter(
                block.project_bn.bias.data
            )
            self.register_buffer(
                prefix + "project_bn_mean", block.project_bn.running_mean
            )
            self.register_buffer(
                prefix + "project_bn_var", block.project_bn.running_var
            )

        # Copy final layer parameters
        self.params["conv_final_weight"] = nn.Parameter(
            original_model.conv_final.weight.data
        )
        self.params["bn_final_weight"] = nn.Parameter(
            original_model.bn_final.weight.data
        )
        self.params["bn_final_bias"] = nn.Parameter(original_model.bn_final.bias.data)
        self.register_buffer("bn_final_mean", original_model.bn_final.running_mean)
        self.register_buffer("bn_final_var", original_model.bn_final.running_var)

        self.params["fc_weight"] = nn.Parameter(original_model.fc.weight.data)
        self.params["fc_bias"] = nn.Parameter(original_model.fc.bias.data)

    def forward(self, x, fn=module_fn):
        params = {
            **dict(self.params),
            **{k: v for k, v in self._buffers.items() if v is not None},
        }
        return fn(x, params, self.training)


batch_size = 2
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]


def get_init_inputs():
    return [num_classes]

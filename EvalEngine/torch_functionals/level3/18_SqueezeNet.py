import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, params: nn.ParameterDict) -> torch.Tensor:
    """
    Implements the SqueezeNet forward pass.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        params (nn.ParameterDict): Dictionary of parameters

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, num_classes)
    """
    # First conv + pool
    x = F.conv2d(x, params["conv1_weight"], params["conv1_bias"], stride=2)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    def fire_module_fn(
        x,
        squeeze_weight,
        squeeze_bias,
        expand1x1_weight,
        expand1x1_bias,
        expand3x3_weight,
        expand3x3_bias,
    ):
        """
        Functional version of FireModule
        """
        x = F.conv2d(x, squeeze_weight, squeeze_bias)
        x = F.relu(x)

        x1 = F.conv2d(x, expand1x1_weight, expand1x1_bias)
        x1 = F.relu(x1)

        x2 = F.conv2d(x, expand3x3_weight, expand3x3_bias, padding=1)
        x2 = F.relu(x2)

        return torch.cat([x1, x2], 1)

    # Fire modules
    x = fire_module_fn(
        x,
        params["fire1_squeeze_weight"],
        params["fire1_squeeze_bias"],
        params["fire1_expand1x1_weight"],
        params["fire1_expand1x1_bias"],
        params["fire1_expand3x3_weight"],
        params["fire1_expand3x3_bias"],
    )

    x = fire_module_fn(
        x,
        params["fire2_squeeze_weight"],
        params["fire2_squeeze_bias"],
        params["fire2_expand1x1_weight"],
        params["fire2_expand1x1_bias"],
        params["fire2_expand3x3_weight"],
        params["fire2_expand3x3_bias"],
    )

    x = fire_module_fn(
        x,
        params["fire3_squeeze_weight"],
        params["fire3_squeeze_bias"],
        params["fire3_expand1x1_weight"],
        params["fire3_expand1x1_bias"],
        params["fire3_expand3x3_weight"],
        params["fire3_expand3x3_bias"],
    )

    x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    x = fire_module_fn(
        x,
        params["fire4_squeeze_weight"],
        params["fire4_squeeze_bias"],
        params["fire4_expand1x1_weight"],
        params["fire4_expand1x1_bias"],
        params["fire4_expand3x3_weight"],
        params["fire4_expand3x3_bias"],
    )

    x = fire_module_fn(
        x,
        params["fire5_squeeze_weight"],
        params["fire5_squeeze_bias"],
        params["fire5_expand1x1_weight"],
        params["fire5_expand1x1_bias"],
        params["fire5_expand3x3_weight"],
        params["fire5_expand3x3_bias"],
    )

    x = fire_module_fn(
        x,
        params["fire6_squeeze_weight"],
        params["fire6_squeeze_bias"],
        params["fire6_expand1x1_weight"],
        params["fire6_expand1x1_bias"],
        params["fire6_expand3x3_weight"],
        params["fire6_expand3x3_bias"],
    )

    x = fire_module_fn(
        x,
        params["fire7_squeeze_weight"],
        params["fire7_squeeze_bias"],
        params["fire7_expand1x1_weight"],
        params["fire7_expand1x1_bias"],
        params["fire7_expand3x3_weight"],
        params["fire7_expand3x3_bias"],
    )

    x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    x = fire_module_fn(
        x,
        params["fire8_squeeze_weight"],
        params["fire8_squeeze_bias"],
        params["fire8_expand1x1_weight"],
        params["fire8_expand1x1_bias"],
        params["fire8_expand3x3_weight"],
        params["fire8_expand3x3_bias"],
    )

    # Classifier
    x = F.conv2d(x, params["classifier_weight"], params["classifier_bias"])
    x = F.relu(x)
    x = F.adaptive_avg_pool2d(x, (1, 1))

    return torch.flatten(x, 1)


class _FireModule(nn.Module):
    """Temporary FireModule class just for parameter initialization"""

    def __init__(
        self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels
    ):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, kernel_size=3, padding=1
        )


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        self.params = nn.ParameterDict()

        # First conv
        conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.params["conv1_weight"] = nn.Parameter(conv1.weight.data.clone())
        self.params["conv1_bias"] = nn.Parameter(conv1.bias.data.clone())

        # Fire modules
        fire_configs = [
            (96, 16, 64, 64),  # fire1
            (128, 16, 64, 64),  # fire2
            (128, 32, 128, 128),  # fire3
            (256, 32, 128, 128),  # fire4
            (256, 48, 192, 192),  # fire5
            (384, 48, 192, 192),  # fire6
            (384, 64, 256, 256),  # fire7
            (512, 64, 256, 256),  # fire8
        ]

        for i, (in_c, sq_c, ex1_c, ex3_c) in enumerate(fire_configs, 1):
            fire = _FireModule(in_c, sq_c, ex1_c, ex3_c)

            self.params[f"fire{i}_squeeze_weight"] = nn.Parameter(
                fire.squeeze.weight.data.clone()
            )
            self.params[f"fire{i}_squeeze_bias"] = nn.Parameter(
                fire.squeeze.bias.data.clone()
            )

            self.params[f"fire{i}_expand1x1_weight"] = nn.Parameter(
                fire.expand1x1.weight.data.clone()
            )
            self.params[f"fire{i}_expand1x1_bias"] = nn.Parameter(
                fire.expand1x1.bias.data.clone()
            )

            self.params[f"fire{i}_expand3x3_weight"] = nn.Parameter(
                fire.expand3x3.weight.data.clone()
            )
            self.params[f"fire{i}_expand3x3_bias"] = nn.Parameter(
                fire.expand3x3.bias.data.clone()
            )

        # Classifier
        classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        self.params["classifier_weight"] = nn.Parameter(classifier.weight.data.clone())
        self.params["classifier_bias"] = nn.Parameter(classifier.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params)


# Test code
batch_size = 1
input_channels = 3
height = 224
width = 224
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [num_classes]

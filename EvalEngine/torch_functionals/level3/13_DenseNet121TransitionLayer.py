import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    params: nn.ParameterDict,
    is_training: bool,
) -> torch.Tensor:
    """
    Implements the DenseNet121 transition layer.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, num_input_features, height, width)
        params (nn.ParameterDict): Dictionary of parameters
        is_training (bool): Whether to use training mode

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, num_output_features, height//2, width//2)
    """
    x = F.batch_norm(
        x,
        params["batchnorm_running_mean"],
        params["batchnorm_running_var"],
        weight=params["batchnorm_weight"],
        bias=params["batchnorm_bias"],
        training=is_training,
    )
    x = F.relu(x)
    x = F.conv2d(x, params["conv_weight"], bias=None)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x


class Model(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(Model, self).__init__()

        self.params = nn.ParameterDict()

        bn = nn.BatchNorm2d(num_input_features)
        self.params["batchnorm_weight"] = nn.Parameter(bn.weight.data.clone())
        self.params["batchnorm_bias"] = nn.Parameter(bn.bias.data.clone())
        self.params["batchnorm_running_mean"] = nn.Parameter(
            bn.running_mean.data.clone()
        )
        self.params["batchnorm_running_var"] = nn.Parameter(bn.running_var.data.clone())

        conv = nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, bias=False
        )
        self.params["conv_weight"] = nn.Parameter(conv.weight.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


batch_size = 10
num_input_features = 32
num_output_features = 64
height, width = 224, 224


def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]


def get_init_inputs():
    return [num_input_features, num_output_features]

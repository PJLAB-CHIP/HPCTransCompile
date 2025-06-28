import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    params: nn.ParameterDict,
    is_training: bool,
) -> torch.Tensor:
    """
    Implements the DenseNet121 dense block.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, num_input_features, height, width)
        params (nn.ParameterDict): Dictionary of parameters
        is_training (bool): Whether to use training mode

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, num_output_features, height, width)
    """

    def layer_fn(
        x,
        bn_weight,
        bn_bias,
        bn_mean,
        bn_var,
        conv_weight,
        is_training,
    ):
        """
        Functional version of a single layer with BatchNorm, ReLU, Conv2D
        """
        x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=is_training)
        x = F.relu(x)
        x = F.conv2d(x, conv_weight, bias=None, padding=1)
        x = F.dropout(x, p=0.0, training=is_training)
        return x

    features = [x]
    for i in range(len(params["bn_weights"])):
        new_feature = layer_fn(
            x,
            params["bn_weights"][i],
            params["bn_biases"][i],
            params["bn_means"][i],
            params["bn_vars"][i],
            params["conv_weights"][i],
            is_training,
        )
        features.append(new_feature)
        x = torch.cat(features, 1)
    return x


class Model(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block
        """
        super(Model, self).__init__()

        params = {
            "bn_weights": [],
            "bn_biases": [],
            "bn_means": [],
            "bn_vars": [],
            "conv_weights": [],
        }
        self.params = nn.ParameterDict()

        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate

            # Create temporary modules to get initialized parameters
            bn = nn.BatchNorm2d(in_features)
            conv = nn.Conv2d(
                in_features, growth_rate, kernel_size=3, padding=1, bias=False
            )

            # Store parameters
            params["bn_weights"].append(bn.weight.data.clone())
            params["bn_biases"].append(bn.bias.data.clone())
            params["bn_means"].append(bn.running_mean.data.clone())
            params["bn_vars"].append(bn.running_var.data.clone())
            params["conv_weights"].append(conv.weight.data.clone())

        # Convert to Parameters
        self.params["bn_weights"] = nn.ParameterList(
            [nn.Parameter(p) for p in params["bn_weights"]]
        )
        self.params["bn_biases"] = nn.ParameterList(
            [nn.Parameter(p) for p in params["bn_biases"]]
        )
        self.params["bn_means"] = nn.ParameterList(
            [nn.Parameter(p) for p in params["bn_means"]]
        )
        self.params["bn_vars"] = nn.ParameterList(
            [nn.Parameter(p) for p in params["bn_vars"]]
        )
        self.params["conv_weights"] = nn.ParameterList(
            [nn.Parameter(p) for p in params["conv_weights"]]
        )

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224


def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]


def get_init_inputs():
    return [num_layers, num_input_features, growth_rate]

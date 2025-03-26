import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    params: nn.ParameterDict,
    is_training: bool,
) -> torch.Tensor:
    """
    Implements the DenseNet121 module.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        params (nn.ParameterDict): Dictionary of parameters
        is_training (bool): Whether to use training mode

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, num_classes)
    """
    # Initial features
    x = F.conv2d(x, params["features_conv_weight"], bias=None, stride=2, padding=3)
    x = F.batch_norm(
        x,
        params["features_bn_mean"],
        params["features_bn_var"],
        params["features_bn_weight"],
        params["features_bn_bias"],
        training=is_training,
    )
    x = F.relu(x, inplace=True)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    def dense_layer_fn(
        x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight, is_training
    ):
        """
        Functional version of a single dense layer
        """
        x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=is_training)
        x = F.relu(x, inplace=True)
        x = F.conv2d(x, conv_weight, bias=None, stride=1, padding=1)
        x = F.dropout(x, p=0.0, training=is_training)
        return x

    def transition_layer_fn(
        x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight, is_training
    ):
        """
        Functional version of transition layer
        """
        x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=is_training)
        x = F.relu(x, inplace=True)
        x = F.conv2d(x, conv_weight, bias=None)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

    # Dense blocks and transitions
    for i in range(4):  # 4 dense blocks
        features = [x]
        for j in range(params[f"block{i}_num_layers"]):  # layers per block
            prefix = f"block{i}_layer{j}_"
            new_feature = dense_layer_fn(
                x,
                params[prefix + "bn_weight"],
                params[prefix + "bn_bias"],
                params[prefix + "bn_mean"],
                params[prefix + "bn_var"],
                params[prefix + "conv_weight"],
                is_training,
            )
            features.append(new_feature)
            x = torch.cat(features, 1)

        if i != 3:  # Apply transition after all blocks except last
            x = transition_layer_fn(
                x,
                params[f"transition{i}_bn_weight"],
                params[f"transition{i}_bn_bias"],
                params[f"transition{i}_bn_mean"],
                params[f"transition{i}_bn_var"],
                params[f"transition{i}_conv_weight"],
                is_training,
            )

    # Final layers
    x = F.batch_norm(
        x,
        params["final_bn_mean"],
        params["final_bn_var"],
        params["final_bn_weight"],
        params["final_bn_bias"],
        training=is_training,
    )
    x = F.relu(x, inplace=True)
    x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
    x = F.linear(x, params["classifier_weight"], params["classifier_bias"])
    return x


class Model(nn.Module):
    def __init__(self, growth_rate=32, num_classes=1000):
        super(Model, self).__init__()

        self.params = nn.ParameterDict()
        block_layers = [6, 12, 24, 16]

        # Initial features parameters
        conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn = nn.BatchNorm2d(64)
        self.params["features_conv_weight"] = nn.Parameter(conv.weight.data.clone())
        self.params["features_bn_weight"] = nn.Parameter(bn.weight.data.clone())
        self.params["features_bn_bias"] = nn.Parameter(bn.bias.data.clone())
        self.params["features_bn_mean"] = nn.Parameter(bn.running_mean.data.clone())
        self.params["features_bn_var"] = nn.Parameter(bn.running_var.data.clone())

        # Dense blocks parameters
        num_features = 64
        for i, num_layers in enumerate(block_layers):
            self.params[f"block{i}_num_layers"] = num_layers
            for j in range(num_layers):
                in_features = num_features + j * growth_rate
                prefix = f"block{i}_layer{j}_"

                bn = nn.BatchNorm2d(in_features)
                conv = nn.Conv2d(
                    in_features, growth_rate, kernel_size=3, padding=1, bias=False
                )

                self.params[prefix + "bn_weight"] = nn.Parameter(bn.weight.data.clone())
                self.params[prefix + "bn_bias"] = nn.Parameter(bn.bias.data.clone())
                self.params[prefix + "bn_mean"] = nn.Parameter(
                    bn.running_mean.data.clone()
                )
                self.params[prefix + "bn_var"] = nn.Parameter(
                    bn.running_var.data.clone()
                )
                self.params[prefix + "conv_weight"] = nn.Parameter(
                    conv.weight.data.clone()
                )

            num_features = num_features + num_layers * growth_rate

            # Transition layers parameters (except after last block)
            if i != len(block_layers) - 1:
                bn = nn.BatchNorm2d(num_features)
                conv = nn.Conv2d(
                    num_features, num_features // 2, kernel_size=1, bias=False
                )

                self.params[f"transition{i}_bn_weight"] = nn.Parameter(
                    bn.weight.data.clone()
                )
                self.params[f"transition{i}_bn_bias"] = nn.Parameter(
                    bn.bias.data.clone()
                )
                self.params[f"transition{i}_bn_mean"] = nn.Parameter(
                    bn.running_mean.data.clone()
                )
                self.params[f"transition{i}_bn_var"] = nn.Parameter(
                    bn.running_var.data.clone()
                )
                self.params[f"transition{i}_conv_weight"] = nn.Parameter(
                    conv.weight.data.clone()
                )

                num_features = num_features // 2

        # Final layers parameters
        bn = nn.BatchNorm2d(num_features)
        self.params["final_bn_weight"] = nn.Parameter(bn.weight.data.clone())
        self.params["final_bn_bias"] = nn.Parameter(bn.bias.data.clone())
        self.params["final_bn_mean"] = nn.Parameter(bn.running_mean.data.clone())
        self.params["final_bn_var"] = nn.Parameter(bn.running_var.data.clone())

        linear = nn.Linear(num_features, num_classes)
        self.params["classifier_weight"] = nn.Parameter(linear.weight.data.clone())
        self.params["classifier_bias"] = nn.Parameter(linear.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


# Test configurations
batch_size = 10
num_classes = 10
height, width = 224, 224


def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]


def get_init_inputs():
    return [32, num_classes]

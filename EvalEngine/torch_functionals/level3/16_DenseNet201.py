import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, params, is_training):
    """
    Functional version of Model forward pass
    """
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
        x = F.conv2d(x, conv_weight, bias=None, padding=1)
        x = F.dropout(x, p=0.0, training=is_training)
        return x

    def dense_block_fn(x, layer_params, is_training):
        """
        Functional version of DenseBlock
        """
        features = [x]
        for params in layer_params:
            new_feature = dense_layer_fn(x, *params, is_training)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

    def transition_layer_fn(
        x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight, is_training
    ):
        """
        Functional version of TransitionLayer
        """
        x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=is_training)
        x = F.relu(x, inplace=True)
        x = F.conv2d(x, conv_weight, bias=None)  # Removed kernel_size parameter
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

    # Dense blocks and transitions
    for i in range(len(params["dense_blocks"])):
        x = dense_block_fn(x, params["dense_blocks"][i], is_training)
        if i != len(params["dense_blocks"]) - 1:
            x = transition_layer_fn(x, *params["transition_layers"][i], is_training)

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
        num_features = 64
        block_layers = [6, 12, 48, 32]
        device = "cuda"

        # Extract initial features parameters
        conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn = nn.BatchNorm2d(64)
        self.params["features_conv_weight"] = nn.Parameter(conv.weight.data.clone()).to(
            device
        )
        self.params["features_bn_weight"] = nn.Parameter(bn.weight.data.clone()).to(
            device
        )
        self.params["features_bn_bias"] = nn.Parameter(bn.bias.data.clone()).to(device)
        self.params["features_bn_mean"] = nn.Parameter(bn.running_mean.data.clone()).to(
            device
        )
        self.params["features_bn_var"] = nn.Parameter(bn.running_var.data.clone()).to(
            device
        )

        # Extract dense blocks parameters
        self.params["dense_blocks"] = []
        for num_layers in block_layers:
            block_params = []
            for i in range(num_layers):
                in_features = num_features + i * growth_rate
                bn = nn.BatchNorm2d(in_features)
                conv = nn.Conv2d(
                    in_features, growth_rate, kernel_size=3, padding=1, bias=False
                )
                layer_params = [
                    nn.Parameter(bn.weight.data.clone()).to(device),
                    nn.Parameter(bn.bias.data.clone()).to(device),
                    nn.Parameter(bn.running_mean.data.clone()).to(device),
                    nn.Parameter(bn.running_var.data.clone()).to(device),
                    nn.Parameter(conv.weight.data.clone()).to(device),
                ]
                block_params.append(layer_params)
            self.params["dense_blocks"].append(block_params)
            num_features = num_features + num_layers * growth_rate

            # Extract transition layer parameters if not last block
            if len(self.params.get("transition_layers", [])) < len(block_layers) - 1:
                bn = nn.BatchNorm2d(num_features)
                conv = nn.Conv2d(
                    num_features, num_features // 2, kernel_size=1, bias=False
                )
                if "transition_layers" not in self.params:
                    self.params["transition_layers"] = []
                self.params["transition_layers"].append(
                    [
                        nn.Parameter(bn.weight.data.clone()).to(device),
                        nn.Parameter(bn.bias.data.clone()).to(device),
                        nn.Parameter(bn.running_mean.data.clone()).to(device),
                        nn.Parameter(bn.running_var.data.clone()).to(device),
                        nn.Parameter(conv.weight.data.clone()).to(device),
                    ]
                )
                num_features = num_features // 2

        # Extract final layers parameters
        bn = nn.BatchNorm2d(num_features)
        self.params["final_bn_weight"] = nn.Parameter(bn.weight.data.clone()).to(device)
        self.params["final_bn_bias"] = nn.Parameter(bn.bias.data.clone()).to(device)
        self.params["final_bn_mean"] = nn.Parameter(bn.running_mean.data.clone()).to(
            device
        )
        self.params["final_bn_var"] = nn.Parameter(bn.running_var.data.clone()).to(
            device
        )

        linear = nn.Linear(num_features, num_classes)
        self.params["classifier_weight"] = nn.Parameter(linear.weight.data.clone()).to(
            device
        )
        self.params["classifier_bias"] = nn.Parameter(linear.bias.data.clone()).to(
            device
        )

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


batch_size = 10
num_classes = 10
height, width = 224, 224


def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]


def get_init_inputs():
    return [32, num_classes]

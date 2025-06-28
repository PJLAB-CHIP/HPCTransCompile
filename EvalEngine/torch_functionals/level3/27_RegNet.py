import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stage_params: nn.ParameterList,
    fc_weight: torch.Tensor,
    fc_bias: torch.Tensor,
    is_training: bool,
) -> torch.Tensor:
    """
    Implementation of RegNet.

    Args:
        x: Input tensor, shape (batch_size, in_channels, height, width)
        stage_params: List of parameters for each stage
        fc_weight: Weight tensor for the final fully connected layer
        fc_bias: Bias tensor for the final fully connected layer
        is_training: Whether in training mode

    Returns:
        Output tensor, shape (batch_size, out_classes)
    """

    def stage_fn(
        x,
        conv1_weight,
        conv1_bias,
        bn1_weight,
        bn1_bias,
        bn1_mean,
        bn1_var,
        conv2_weight,
        conv2_bias,
        bn2_weight,
        bn2_bias,
        bn2_mean,
        bn2_var,
        is_training,
    ):
        """
        Functional implementation of a single stage block
        """
        x = F.conv2d(x, conv1_weight, conv1_bias, padding=1)
        x = F.batch_norm(
            x, bn1_mean, bn1_var, bn1_weight, bn1_bias, training=is_training
        )
        x = F.relu(x)

        x = F.conv2d(x, conv2_weight, conv2_bias, padding=1)
        x = F.batch_norm(
            x, bn2_mean, bn2_var, bn2_weight, bn2_bias, training=is_training
        )
        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    # Pass through all stages
    for stage_param in stage_params:
        x = stage_fn(x, *stage_param, is_training)

    # Global average pooling
    x = torch.mean(x, dim=[2, 3])

    # Final classification
    x = F.linear(x, fc_weight, fc_bias)
    return x


class Model(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(Model, self).__init__()

        self.stages = stages
        self.block_widths = block_widths

        # Store parameters for each stage
        self.stage_params = nn.ParameterList()
        current_channels = input_channels

        for i in range(stages):
            # Create temporary stage to extract parameters
            stage = nn.Sequential(
                nn.Conv2d(current_channels, block_widths[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(block_widths[i]),
                nn.ReLU(),
                nn.Conv2d(block_widths[i], block_widths[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(block_widths[i]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            # Extract and store parameters for this stage
            stage_params = []

            # First conv + bn
            stage_params.append(nn.Parameter(stage[0].weight.data.clone()))
            stage_params.append(nn.Parameter(stage[0].bias.data.clone()))
            stage_params.append(nn.Parameter(stage[1].weight.data.clone()))
            stage_params.append(nn.Parameter(stage[1].bias.data.clone()))
            stage_params.append(nn.Parameter(stage[1].running_mean.data.clone()))
            stage_params.append(nn.Parameter(stage[1].running_var.data.clone()))

            # Second conv + bn
            stage_params.append(nn.Parameter(stage[3].weight.data.clone()))
            stage_params.append(nn.Parameter(stage[3].bias.data.clone()))
            stage_params.append(nn.Parameter(stage[4].weight.data.clone()))
            stage_params.append(nn.Parameter(stage[4].bias.data.clone()))
            stage_params.append(nn.Parameter(stage[4].running_mean.data.clone()))
            stage_params.append(nn.Parameter(stage[4].running_var.data.clone()))

            self.stage_params.append(nn.ParameterList(stage_params))
            current_channels = block_widths[i]

        # Final fully connected layer parameters
        fc = nn.Linear(block_widths[-1], output_classes)
        self.fc_weight = nn.Parameter(fc.weight.data.clone())
        self.fc_bias = nn.Parameter(fc.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.stage_params, self.fc_weight, self.fc_bias, self.training)


# Test code for the RegNet model
batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10


def get_inputs():
    """Generates random input tensor of shape (batch_size, input_channels, height, width)"""
    return [torch.randn(batch_size, input_channels, image_height, image_width)]


def get_init_inputs():
    """Initializes model parameters"""
    return [input_channels, stages, block_widths, output_classes]

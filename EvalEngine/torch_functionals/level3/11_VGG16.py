import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weights: nn.ParameterList,
    conv_biases: nn.ParameterList,
    fc_weights: nn.ParameterList,
    fc_biases: nn.ParameterList,
    is_training: bool,
) -> torch.Tensor:
    """
    Implements the VGG16 module.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        conv_weights (nn.ParameterList): List of convolutional weights
        conv_biases (nn.ParameterList): List of convolutional biases
        fc_weights (nn.ParameterList): List of fully connected weights
        fc_biases (nn.ParameterList): List of fully connected biases
        is_training (bool): Whether in training mode

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, num_classes)
    """
    # Block 1
    x = F.conv2d(x, conv_weights[0], conv_biases[0], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_weights[1], conv_biases[1], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Block 2
    x = F.conv2d(x, conv_weights[2], conv_biases[2], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_weights[3], conv_biases[3], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Block 3
    x = F.conv2d(x, conv_weights[4], conv_biases[4], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_weights[5], conv_biases[5], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_weights[6], conv_biases[6], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Block 4
    x = F.conv2d(x, conv_weights[7], conv_biases[7], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_weights[8], conv_biases[8], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_weights[9], conv_biases[9], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Block 5
    x = F.conv2d(x, conv_weights[10], conv_biases[10], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_weights[11], conv_biases[11], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_weights[12], conv_biases[12], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Classifier
    x = torch.flatten(x, 1)
    x = F.linear(x, fc_weights[0], fc_biases[0])
    x = F.relu(x)
    x = F.dropout(x, p=0.0, training=is_training)
    x = F.linear(x, fc_weights[1], fc_biases[1])
    x = F.relu(x)
    x = F.dropout(x, p=0.0, training=is_training)
    x = F.linear(x, fc_weights[2], fc_biases[2])

    return x


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        # Extract convolutional parameters
        self.conv_weights = nn.ParameterList()
        self.conv_biases = nn.ParameterList()

        # Block 1
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        # Block 2
        conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        # Block 3
        conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        # Block 4
        conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        # Block 5
        conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_weights.append(nn.Parameter(conv.weight.data.clone()))
        self.conv_biases.append(nn.Parameter(conv.bias.data.clone()))

        # Extract fully connected parameters
        self.fc_weights = nn.ParameterList()
        self.fc_biases = nn.ParameterList()

        fc = nn.Linear(512 * 7 * 7, 4096)
        self.fc_weights.append(nn.Parameter(fc.weight.data.clone()))
        self.fc_biases.append(nn.Parameter(fc.bias.data.clone()))

        fc = nn.Linear(4096, 4096)
        self.fc_weights.append(nn.Parameter(fc.weight.data.clone()))
        self.fc_biases.append(nn.Parameter(fc.bias.data.clone()))

        fc = nn.Linear(4096, num_classes)
        self.fc_weights.append(nn.Parameter(fc.weight.data.clone()))
        self.fc_biases.append(nn.Parameter(fc.bias.data.clone()))

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_weights,
            self.conv_biases,
            self.fc_weights,
            self.fc_biases,
            self.training,
        )


# Test code
batch_size = 10
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]


def get_init_inputs():
    return [num_classes]

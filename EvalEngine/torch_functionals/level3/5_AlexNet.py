import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, params: nn.ParameterDict, is_training: bool
) -> torch.Tensor:
    """
    Implements the AlexNet architecture.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        params (nn.ParameterDict): Dictionary containing model parameters
        is_training (bool): Whether in training mode

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes)
    """
    # First conv block
    x = F.conv2d(x, params["conv1_weight"], params["conv1_bias"], stride=4, padding=2)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)

    # Second conv block
    x = F.conv2d(x, params["conv2_weight"], params["conv2_bias"], padding=2)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)

    # Third conv block
    x = F.conv2d(x, params["conv3_weight"], params["conv3_bias"], padding=1)
    x = F.relu(x)

    # Fourth conv block
    x = F.conv2d(x, params["conv4_weight"], params["conv4_bias"], padding=1)
    x = F.relu(x)

    # Fifth conv block
    x = F.conv2d(x, params["conv5_weight"], params["conv5_bias"], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)

    # Flatten
    x = torch.flatten(x, 1)

    # FC layers
    x = F.linear(x, params["fc1_weight"], params["fc1_bias"])
    x = F.relu(x)
    x = F.dropout(x, p=0.0, training=is_training)

    x = F.linear(x, params["fc2_weight"], params["fc2_bias"])
    x = F.relu(x)
    x = F.dropout(x, p=0.0, training=is_training)

    x = F.linear(x, params["fc3_weight"], params["fc3_bias"])

    return x


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(Model, self).__init__()

        self.params = nn.ParameterDict()

        # Extract conv1 parameters
        conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.params["conv1_weight"] = nn.Parameter(conv1.weight.data.clone())
        self.params["conv1_bias"] = nn.Parameter(conv1.bias.data.clone())

        # Extract conv2 parameters
        conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.params["conv2_weight"] = nn.Parameter(conv2.weight.data.clone())
        self.params["conv2_bias"] = nn.Parameter(conv2.bias.data.clone())

        # Extract conv3 parameters
        conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.params["conv3_weight"] = nn.Parameter(conv3.weight.data.clone())
        self.params["conv3_bias"] = nn.Parameter(conv3.bias.data.clone())

        # Extract conv4 parameters
        conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.params["conv4_weight"] = nn.Parameter(conv4.weight.data.clone())
        self.params["conv4_bias"] = nn.Parameter(conv4.bias.data.clone())

        # Extract conv5 parameters
        conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.params["conv5_weight"] = nn.Parameter(conv5.weight.data.clone())
        self.params["conv5_bias"] = nn.Parameter(conv5.bias.data.clone())

        # Extract fc1 parameters
        fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.params["fc1_weight"] = nn.Parameter(fc1.weight.data.clone())
        self.params["fc1_bias"] = nn.Parameter(fc1.bias.data.clone())

        # Extract fc2 parameters
        fc2 = nn.Linear(4096, 4096)
        self.params["fc2_weight"] = nn.Parameter(fc2.weight.data.clone())
        self.params["fc2_bias"] = nn.Parameter(fc2.bias.data.clone())

        # Extract fc3 parameters
        fc3 = nn.Linear(4096, num_classes)
        self.params["fc3_weight"] = nn.Parameter(fc3.weight.data.clone())
        self.params["fc3_bias"] = nn.Parameter(fc3.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


# Test code
batch_size = 10
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]


def get_init_inputs():
    return [num_classes]

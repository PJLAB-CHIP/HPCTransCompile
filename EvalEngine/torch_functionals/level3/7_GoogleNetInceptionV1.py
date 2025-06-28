import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    def __init__(
        self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj
    ):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
        )

        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )

        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )


def module_fn(
    x: torch.Tensor, params: nn.ParameterDict, is_training: bool
) -> torch.Tensor:
    """
    Implements the GoogleNet Inception module.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        params (nn.ParameterDict): Parameter dictionary containing the model parameters
        is_training (bool): Whether the model is in training mode

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, out_channels, height, width)
    """
    # Initial convolutions
    x = F.conv2d(x, params["conv1_w"], params["conv1_b"], stride=2, padding=3)
    x = F.relu(x)
    x = F.max_pool2d(x, 3, stride=2, padding=1)

    x = F.conv2d(x, params["conv2_w"], params["conv2_b"])
    x = F.relu(x)

    x = F.conv2d(x, params["conv3_w"], params["conv3_b"], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, 3, stride=2, padding=1)

    def inception_module_fn(
        x,
        conv1x1_w,
        conv1x1_b,
        conv3x3_reduce_w,
        conv3x3_reduce_b,
        conv3x3_w,
        conv3x3_b,
        conv5x5_reduce_w,
        conv5x5_reduce_b,
        conv5x5_w,
        conv5x5_b,
        pool_proj_w,
        pool_proj_b,
    ):
        # 1x1 branch
        branch1x1 = F.conv2d(x, conv1x1_w, conv1x1_b)

        # 3x3 branch
        branch3x3 = F.conv2d(x, conv3x3_reduce_w, conv3x3_reduce_b)
        branch3x3 = F.conv2d(branch3x3, conv3x3_w, conv3x3_b, padding=1)

        # 5x5 branch
        branch5x5 = F.conv2d(x, conv5x5_reduce_w, conv5x5_reduce_b)
        branch5x5 = F.conv2d(branch5x5, conv5x5_w, conv5x5_b, padding=2)

        # Pool branch
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = F.conv2d(branch_pool, pool_proj_w, pool_proj_b)

        return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)

    # Inception modules
    x = inception_module_fn(
        x,
        params["3a_1x1_w"],
        params["3a_1x1_b"],
        params["3a_3x3_reduce_w"],
        params["3a_3x3_reduce_b"],
        params["3a_3x3_w"],
        params["3a_3x3_b"],
        params["3a_5x5_reduce_w"],
        params["3a_5x5_reduce_b"],
        params["3a_5x5_w"],
        params["3a_5x5_b"],
        params["3a_pool_proj_w"],
        params["3a_pool_proj_b"],
    )

    x = inception_module_fn(
        x,
        params["3b_1x1_w"],
        params["3b_1x1_b"],
        params["3b_3x3_reduce_w"],
        params["3b_3x3_reduce_b"],
        params["3b_3x3_w"],
        params["3b_3x3_b"],
        params["3b_5x5_reduce_w"],
        params["3b_5x5_reduce_b"],
        params["3b_5x5_w"],
        params["3b_5x5_b"],
        params["3b_pool_proj_w"],
        params["3b_pool_proj_b"],
    )

    x = F.max_pool2d(x, 3, stride=2, padding=1)

    x = inception_module_fn(
        x,
        params["4a_1x1_w"],
        params["4a_1x1_b"],
        params["4a_3x3_reduce_w"],
        params["4a_3x3_reduce_b"],
        params["4a_3x3_w"],
        params["4a_3x3_b"],
        params["4a_5x5_reduce_w"],
        params["4a_5x5_reduce_b"],
        params["4a_5x5_w"],
        params["4a_5x5_b"],
        params["4a_pool_proj_w"],
        params["4a_pool_proj_b"],
    )

    x = inception_module_fn(
        x,
        params["4b_1x1_w"],
        params["4b_1x1_b"],
        params["4b_3x3_reduce_w"],
        params["4b_3x3_reduce_b"],
        params["4b_3x3_w"],
        params["4b_3x3_b"],
        params["4b_5x5_reduce_w"],
        params["4b_5x5_reduce_b"],
        params["4b_5x5_w"],
        params["4b_5x5_b"],
        params["4b_pool_proj_w"],
        params["4b_pool_proj_b"],
    )

    x = inception_module_fn(
        x,
        params["4c_1x1_w"],
        params["4c_1x1_b"],
        params["4c_3x3_reduce_w"],
        params["4c_3x3_reduce_b"],
        params["4c_3x3_w"],
        params["4c_3x3_b"],
        params["4c_5x5_reduce_w"],
        params["4c_5x5_reduce_b"],
        params["4c_5x5_w"],
        params["4c_5x5_b"],
        params["4c_pool_proj_w"],
        params["4c_pool_proj_b"],
    )

    x = inception_module_fn(
        x,
        params["4d_1x1_w"],
        params["4d_1x1_b"],
        params["4d_3x3_reduce_w"],
        params["4d_3x3_reduce_b"],
        params["4d_3x3_w"],
        params["4d_3x3_b"],
        params["4d_5x5_reduce_w"],
        params["4d_5x5_reduce_b"],
        params["4d_5x5_w"],
        params["4d_5x5_b"],
        params["4d_pool_proj_w"],
        params["4d_pool_proj_b"],
    )

    x = inception_module_fn(
        x,
        params["4e_1x1_w"],
        params["4e_1x1_b"],
        params["4e_3x3_reduce_w"],
        params["4e_3x3_reduce_b"],
        params["4e_3x3_w"],
        params["4e_3x3_b"],
        params["4e_5x5_reduce_w"],
        params["4e_5x5_reduce_b"],
        params["4e_5x5_w"],
        params["4e_5x5_b"],
        params["4e_pool_proj_w"],
        params["4e_pool_proj_b"],
    )

    x = F.max_pool2d(x, 3, stride=2, padding=1)

    x = inception_module_fn(
        x,
        params["5a_1x1_w"],
        params["5a_1x1_b"],
        params["5a_3x3_reduce_w"],
        params["5a_3x3_reduce_b"],
        params["5a_3x3_w"],
        params["5a_3x3_b"],
        params["5a_5x5_reduce_w"],
        params["5a_5x5_reduce_b"],
        params["5a_5x5_w"],
        params["5a_5x5_b"],
        params["5a_pool_proj_w"],
        params["5a_pool_proj_b"],
    )

    x = inception_module_fn(
        x,
        params["5b_1x1_w"],
        params["5b_1x1_b"],
        params["5b_3x3_reduce_w"],
        params["5b_3x3_reduce_b"],
        params["5b_3x3_w"],
        params["5b_3x3_b"],
        params["5b_5x5_reduce_w"],
        params["5b_5x5_reduce_b"],
        params["5b_5x5_w"],
        params["5b_5x5_b"],
        params["5b_pool_proj_w"],
        params["5b_pool_proj_b"],
    )

    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = F.dropout(x, p=0.0, training=is_training)
    x = F.linear(x, params["fc_w"], params["fc_b"])

    return x


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        self.params = nn.ParameterDict()

        # Initial convolutions
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.params["conv1_w"] = nn.Parameter(conv1.weight.data.clone())
        self.params["conv1_b"] = nn.Parameter(conv1.bias.data.clone())

        conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.params["conv2_w"] = nn.Parameter(conv2.weight.data.clone())
        self.params["conv2_b"] = nn.Parameter(conv2.bias.data.clone())

        conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.params["conv3_w"] = nn.Parameter(conv3.weight.data.clone())
        self.params["conv3_b"] = nn.Parameter(conv3.bias.data.clone())

        # Inception 3a
        inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.params["3a_1x1_w"] = nn.Parameter(
            inception3a.branch1x1.weight.data.clone()
        )
        self.params["3a_1x1_b"] = nn.Parameter(inception3a.branch1x1.bias.data.clone())
        self.params["3a_3x3_reduce_w"] = nn.Parameter(
            inception3a.branch3x3[0].weight.data.clone()
        )
        self.params["3a_3x3_reduce_b"] = nn.Parameter(
            inception3a.branch3x3[0].bias.data.clone()
        )
        self.params["3a_3x3_w"] = nn.Parameter(
            inception3a.branch3x3[1].weight.data.clone()
        )
        self.params["3a_3x3_b"] = nn.Parameter(
            inception3a.branch3x3[1].bias.data.clone()
        )
        self.params["3a_5x5_reduce_w"] = nn.Parameter(
            inception3a.branch5x5[0].weight.data.clone()
        )
        self.params["3a_5x5_reduce_b"] = nn.Parameter(
            inception3a.branch5x5[0].bias.data.clone()
        )
        self.params["3a_5x5_w"] = nn.Parameter(
            inception3a.branch5x5[1].weight.data.clone()
        )
        self.params["3a_5x5_b"] = nn.Parameter(
            inception3a.branch5x5[1].bias.data.clone()
        )
        self.params["3a_pool_proj_w"] = nn.Parameter(
            inception3a.branch_pool[1].weight.data.clone()
        )
        self.params["3a_pool_proj_b"] = nn.Parameter(
            inception3a.branch_pool[1].bias.data.clone()
        )

        # Inception 3b
        inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.params["3b_1x1_w"] = nn.Parameter(
            inception3b.branch1x1.weight.data.clone()
        )
        self.params["3b_1x1_b"] = nn.Parameter(inception3b.branch1x1.bias.data.clone())
        self.params["3b_3x3_reduce_w"] = nn.Parameter(
            inception3b.branch3x3[0].weight.data.clone()
        )
        self.params["3b_3x3_reduce_b"] = nn.Parameter(
            inception3b.branch3x3[0].bias.data.clone()
        )
        self.params["3b_3x3_w"] = nn.Parameter(
            inception3b.branch3x3[1].weight.data.clone()
        )
        self.params["3b_3x3_b"] = nn.Parameter(
            inception3b.branch3x3[1].bias.data.clone()
        )
        self.params["3b_5x5_reduce_w"] = nn.Parameter(
            inception3b.branch5x5[0].weight.data.clone()
        )
        self.params["3b_5x5_reduce_b"] = nn.Parameter(
            inception3b.branch5x5[0].bias.data.clone()
        )
        self.params["3b_5x5_w"] = nn.Parameter(
            inception3b.branch5x5[1].weight.data.clone()
        )
        self.params["3b_5x5_b"] = nn.Parameter(
            inception3b.branch5x5[1].bias.data.clone()
        )
        self.params["3b_pool_proj_w"] = nn.Parameter(
            inception3b.branch_pool[1].weight.data.clone()
        )
        self.params["3b_pool_proj_b"] = nn.Parameter(
            inception3b.branch_pool[1].bias.data.clone()
        )

        # Inception 4a
        inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.params["4a_1x1_w"] = nn.Parameter(
            inception4a.branch1x1.weight.data.clone()
        )
        self.params["4a_1x1_b"] = nn.Parameter(inception4a.branch1x1.bias.data.clone())
        self.params["4a_3x3_reduce_w"] = nn.Parameter(
            inception4a.branch3x3[0].weight.data.clone()
        )
        self.params["4a_3x3_reduce_b"] = nn.Parameter(
            inception4a.branch3x3[0].bias.data.clone()
        )
        self.params["4a_3x3_w"] = nn.Parameter(
            inception4a.branch3x3[1].weight.data.clone()
        )
        self.params["4a_3x3_b"] = nn.Parameter(
            inception4a.branch3x3[1].bias.data.clone()
        )
        self.params["4a_5x5_reduce_w"] = nn.Parameter(
            inception4a.branch5x5[0].weight.data.clone()
        )
        self.params["4a_5x5_reduce_b"] = nn.Parameter(
            inception4a.branch5x5[0].bias.data.clone()
        )
        self.params["4a_5x5_w"] = nn.Parameter(
            inception4a.branch5x5[1].weight.data.clone()
        )
        self.params["4a_5x5_b"] = nn.Parameter(
            inception4a.branch5x5[1].bias.data.clone()
        )
        self.params["4a_pool_proj_w"] = nn.Parameter(
            inception4a.branch_pool[1].weight.data.clone()
        )
        self.params["4a_pool_proj_b"] = nn.Parameter(
            inception4a.branch_pool[1].bias.data.clone()
        )

        # Inception 4b
        inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.params["4b_1x1_w"] = nn.Parameter(
            inception4b.branch1x1.weight.data.clone()
        )
        self.params["4b_1x1_b"] = nn.Parameter(inception4b.branch1x1.bias.data.clone())
        self.params["4b_3x3_reduce_w"] = nn.Parameter(
            inception4b.branch3x3[0].weight.data.clone()
        )
        self.params["4b_3x3_reduce_b"] = nn.Parameter(
            inception4b.branch3x3[0].bias.data.clone()
        )
        self.params["4b_3x3_w"] = nn.Parameter(
            inception4b.branch3x3[1].weight.data.clone()
        )
        self.params["4b_3x3_b"] = nn.Parameter(
            inception4b.branch3x3[1].bias.data.clone()
        )
        self.params["4b_5x5_reduce_w"] = nn.Parameter(
            inception4b.branch5x5[0].weight.data.clone()
        )
        self.params["4b_5x5_reduce_b"] = nn.Parameter(
            inception4b.branch5x5[0].bias.data.clone()
        )
        self.params["4b_5x5_w"] = nn.Parameter(
            inception4b.branch5x5[1].weight.data.clone()
        )
        self.params["4b_5x5_b"] = nn.Parameter(
            inception4b.branch5x5[1].bias.data.clone()
        )
        self.params["4b_pool_proj_w"] = nn.Parameter(
            inception4b.branch_pool[1].weight.data.clone()
        )
        self.params["4b_pool_proj_b"] = nn.Parameter(
            inception4b.branch_pool[1].bias.data.clone()
        )

        # Inception 4c
        inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.params["4c_1x1_w"] = nn.Parameter(
            inception4c.branch1x1.weight.data.clone()
        )
        self.params["4c_1x1_b"] = nn.Parameter(inception4c.branch1x1.bias.data.clone())
        self.params["4c_3x3_reduce_w"] = nn.Parameter(
            inception4c.branch3x3[0].weight.data.clone()
        )
        self.params["4c_3x3_reduce_b"] = nn.Parameter(
            inception4c.branch3x3[0].bias.data.clone()
        )
        self.params["4c_3x3_w"] = nn.Parameter(
            inception4c.branch3x3[1].weight.data.clone()
        )
        self.params["4c_3x3_b"] = nn.Parameter(
            inception4c.branch3x3[1].bias.data.clone()
        )
        self.params["4c_5x5_reduce_w"] = nn.Parameter(
            inception4c.branch5x5[0].weight.data.clone()
        )
        self.params["4c_5x5_reduce_b"] = nn.Parameter(
            inception4c.branch5x5[0].bias.data.clone()
        )
        self.params["4c_5x5_w"] = nn.Parameter(
            inception4c.branch5x5[1].weight.data.clone()
        )
        self.params["4c_5x5_b"] = nn.Parameter(
            inception4c.branch5x5[1].bias.data.clone()
        )
        self.params["4c_pool_proj_w"] = nn.Parameter(
            inception4c.branch_pool[1].weight.data.clone()
        )
        self.params["4c_pool_proj_b"] = nn.Parameter(
            inception4c.branch_pool[1].bias.data.clone()
        )

        # Inception 4d
        inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.params["4d_1x1_w"] = nn.Parameter(
            inception4d.branch1x1.weight.data.clone()
        )
        self.params["4d_1x1_b"] = nn.Parameter(inception4d.branch1x1.bias.data.clone())
        self.params["4d_3x3_reduce_w"] = nn.Parameter(
            inception4d.branch3x3[0].weight.data.clone()
        )
        self.params["4d_3x3_reduce_b"] = nn.Parameter(
            inception4d.branch3x3[0].bias.data.clone()
        )
        self.params["4d_3x3_w"] = nn.Parameter(
            inception4d.branch3x3[1].weight.data.clone()
        )
        self.params["4d_3x3_b"] = nn.Parameter(
            inception4d.branch3x3[1].bias.data.clone()
        )
        self.params["4d_5x5_reduce_w"] = nn.Parameter(
            inception4d.branch5x5[0].weight.data.clone()
        )
        self.params["4d_5x5_reduce_b"] = nn.Parameter(
            inception4d.branch5x5[0].bias.data.clone()
        )
        self.params["4d_5x5_w"] = nn.Parameter(
            inception4d.branch5x5[1].weight.data.clone()
        )
        self.params["4d_5x5_b"] = nn.Parameter(
            inception4d.branch5x5[1].bias.data.clone()
        )
        self.params["4d_pool_proj_w"] = nn.Parameter(
            inception4d.branch_pool[1].weight.data.clone()
        )
        self.params["4d_pool_proj_b"] = nn.Parameter(
            inception4d.branch_pool[1].bias.data.clone()
        )

        # Inception 4e
        inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.params["4e_1x1_w"] = nn.Parameter(
            inception4e.branch1x1.weight.data.clone()
        )
        self.params["4e_1x1_b"] = nn.Parameter(inception4e.branch1x1.bias.data.clone())
        self.params["4e_3x3_reduce_w"] = nn.Parameter(
            inception4e.branch3x3[0].weight.data.clone()
        )
        self.params["4e_3x3_reduce_b"] = nn.Parameter(
            inception4e.branch3x3[0].bias.data.clone()
        )
        self.params["4e_3x3_w"] = nn.Parameter(
            inception4e.branch3x3[1].weight.data.clone()
        )
        self.params["4e_3x3_b"] = nn.Parameter(
            inception4e.branch3x3[1].bias.data.clone()
        )
        self.params["4e_5x5_reduce_w"] = nn.Parameter(
            inception4e.branch5x5[0].weight.data.clone()
        )
        self.params["4e_5x5_reduce_b"] = nn.Parameter(
            inception4e.branch5x5[0].bias.data.clone()
        )
        self.params["4e_5x5_w"] = nn.Parameter(
            inception4e.branch5x5[1].weight.data.clone()
        )
        self.params["4e_5x5_b"] = nn.Parameter(
            inception4e.branch5x5[1].bias.data.clone()
        )
        self.params["4e_pool_proj_w"] = nn.Parameter(
            inception4e.branch_pool[1].weight.data.clone()
        )
        self.params["4e_pool_proj_b"] = nn.Parameter(
            inception4e.branch_pool[1].bias.data.clone()
        )

        # Inception 5a
        inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.params["5a_1x1_w"] = nn.Parameter(
            inception5a.branch1x1.weight.data.clone()
        )
        self.params["5a_1x1_b"] = nn.Parameter(inception5a.branch1x1.bias.data.clone())
        self.params["5a_3x3_reduce_w"] = nn.Parameter(
            inception5a.branch3x3[0].weight.data.clone()
        )
        self.params["5a_3x3_reduce_b"] = nn.Parameter(
            inception5a.branch3x3[0].bias.data.clone()
        )
        self.params["5a_3x3_w"] = nn.Parameter(
            inception5a.branch3x3[1].weight.data.clone()
        )
        self.params["5a_3x3_b"] = nn.Parameter(
            inception5a.branch3x3[1].bias.data.clone()
        )
        self.params["5a_5x5_reduce_w"] = nn.Parameter(
            inception5a.branch5x5[0].weight.data.clone()
        )
        self.params["5a_5x5_reduce_b"] = nn.Parameter(
            inception5a.branch5x5[0].bias.data.clone()
        )
        self.params["5a_5x5_w"] = nn.Parameter(
            inception5a.branch5x5[1].weight.data.clone()
        )
        self.params["5a_5x5_b"] = nn.Parameter(
            inception5a.branch5x5[1].bias.data.clone()
        )
        self.params["5a_pool_proj_w"] = nn.Parameter(
            inception5a.branch_pool[1].weight.data.clone()
        )
        self.params["5a_pool_proj_b"] = nn.Parameter(
            inception5a.branch_pool[1].bias.data.clone()
        )

        # Inception 5b
        inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.params["5b_1x1_w"] = nn.Parameter(
            inception5b.branch1x1.weight.data.clone()
        )
        self.params["5b_1x1_b"] = nn.Parameter(inception5b.branch1x1.bias.data.clone())
        self.params["5b_3x3_reduce_w"] = nn.Parameter(
            inception5b.branch3x3[0].weight.data.clone()
        )
        self.params["5b_3x3_reduce_b"] = nn.Parameter(
            inception5b.branch3x3[0].bias.data.clone()
        )
        self.params["5b_3x3_w"] = nn.Parameter(
            inception5b.branch3x3[1].weight.data.clone()
        )
        self.params["5b_3x3_b"] = nn.Parameter(
            inception5b.branch3x3[1].bias.data.clone()
        )
        self.params["5b_5x5_reduce_w"] = nn.Parameter(
            inception5b.branch5x5[0].weight.data.clone()
        )
        self.params["5b_5x5_reduce_b"] = nn.Parameter(
            inception5b.branch5x5[0].bias.data.clone()
        )
        self.params["5b_5x5_w"] = nn.Parameter(
            inception5b.branch5x5[1].weight.data.clone()
        )
        self.params["5b_5x5_b"] = nn.Parameter(
            inception5b.branch5x5[1].bias.data.clone()
        )
        self.params["5b_pool_proj_w"] = nn.Parameter(
            inception5b.branch_pool[1].weight.data.clone()
        )
        self.params["5b_pool_proj_b"] = nn.Parameter(
            inception5b.branch_pool[1].bias.data.clone()
        )

        # Final fully connected layer
        fc = nn.Linear(1024, num_classes)
        self.params["fc_w"] = nn.Parameter(fc.weight.data.clone())
        self.params["fc_b"] = nn.Parameter(fc.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [num_classes]

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(x, conv_weight, conv_bias, multiplier, running_mean, running_var, clamp_min, clamp_max):
    # Conv3d
    x = F.conv3d(x, conv_weight, conv_bias)
    # Multiply by multiplier
    x = x * multiplier
    # InstanceNorm3d
    x = F.instance_norm(x, running_mean=running_mean, running_var=running_var, weight=None, bias=None, use_input_stats=True)
    # Clamp
    x = torch.clamp(x, clamp_min, clamp_max)
    # Multiply by multiplier again
    x = x * multiplier
    # Max along dim=1
    x = torch.max(x, dim=1)[0]
    return x

class Model(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Initialize parameters
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv_bias, -bound, bound)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.multiplier, self.running_mean, self.running_var, self.clamp_min, self.clamp_max)

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]
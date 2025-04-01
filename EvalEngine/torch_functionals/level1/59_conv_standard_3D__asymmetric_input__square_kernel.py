import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel (kernel_size x kernel_size).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
    
    def forward(self, x: torch.Tensor, fn=None) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).
            fn (callable, optional): Functional version of the forward pass. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        if fn is None:
            fn = module_fn
        bias = self.bias_param
        weight = self.weight
        if fn != module_fn:
            if bias is None:
                bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
            weight = weight.detach()  # Convert nn.Parameter to a tensor

        return fn(x, 
                  weight, 
                  bias, 
                  self.stride, 
                  self.padding, 
                  self.dilation, 
                  self.groups)

def module_fn(x, 
            weight, 
            bias_param, 
            stride, 
            padding, 
            dilation, 
            groups):
    return F.conv3d(x, 
            weight, 
            bias_param, 
            stride, 
            padding, 
            dilation, 
            groups)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 12

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width, depth)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization 
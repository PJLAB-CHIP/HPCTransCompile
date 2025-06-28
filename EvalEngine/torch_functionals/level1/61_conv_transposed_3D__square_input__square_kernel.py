import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Define parameters
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias_param', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)
        
    def forward(self, x: torch.Tensor, fn=None) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        if fn is None:
            fn = module_fn
        bias = self.bias_param
        weight = self.weight
        if fn != module_fn:
            if bias is None:
                bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)
            weight = weight.detach()  # Convert nn.Parameter to a tensor
        return fn(x, weight, bias, self.stride, self.padding, self.output_padding, self.groups)

def module_fn(x, weight, bias, stride, padding, output_padding, groups):
    return F.conv_transpose3d(
        x, weight, bias, stride=stride, padding=padding, 
        output_padding=output_padding, groups=groups
    )

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 32
height = 32
width = 32

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization 
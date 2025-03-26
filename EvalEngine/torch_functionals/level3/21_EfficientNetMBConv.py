import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(Model, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv_weight = nn.Parameter(torch.randn(hidden_dim, in_channels, 1, 1))
            self.expand_conv_bn_weight = nn.Parameter(torch.ones(hidden_dim))
            self.expand_conv_bn_bias = nn.Parameter(torch.zeros(hidden_dim))
            self.expand_conv_bn_running_mean = nn.Parameter(torch.zeros(hidden_dim), requires_grad=False)
            self.expand_conv_bn_running_var = nn.Parameter(torch.ones(hidden_dim), requires_grad=False)
        
        self.depthwise_conv_weight = nn.Parameter(torch.randn(hidden_dim, 1, kernel_size, kernel_size))
        self.depthwise_conv_bn_weight = nn.Parameter(torch.ones(hidden_dim))
        self.depthwise_conv_bn_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.depthwise_conv_bn_running_mean = nn.Parameter(torch.zeros(hidden_dim), requires_grad=False)
        self.depthwise_conv_bn_running_var = nn.Parameter(torch.ones(hidden_dim), requires_grad=False)
        
        self.project_conv_weight = nn.Parameter(torch.randn(out_channels, hidden_dim, 1, 1))
        self.project_conv_bn_weight = nn.Parameter(torch.ones(out_channels))
        self.project_conv_bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.project_conv_bn_running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.project_conv_bn_running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
    
    def forward(self, x, fn=None):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :param fn: The function to use for forward pass. Defaults to module_fn.
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        if fn is None:
            fn = module_fn
        return fn(x, self)
        
def module_fn(x, module):
    identity = x
    
    if hasattr(module, 'expand_conv_weight'):
        x = F.conv2d(x, module.expand_conv_weight, bias=None, stride=1, padding=0)
        x = F.batch_norm(x, module.expand_conv_bn_running_mean, module.expand_conv_bn_running_var, 
                          module.expand_conv_bn_weight, module.expand_conv_bn_bias, training=module.training)
        x = F.relu6(x, inplace=True)
    
    x = F.conv2d(x, module.depthwise_conv_weight, bias=None, stride=module.use_residual + 1, 
                 padding=(module.depthwise_conv_weight.shape[2] - 1) // 2, groups=module.depthwise_conv_weight.shape[0])
    x = F.batch_norm(x, module.depthwise_conv_bn_running_mean, module.depthwise_conv_bn_running_var, 
                      module.depthwise_conv_bn_weight, module.depthwise_conv_bn_bias, training=module.training)
    x = F.relu6(x, inplace=True)
    
    x = F.conv2d(x, module.project_conv_weight, bias=None, stride=1, padding=0)
    x = F.batch_norm(x, module.project_conv_bn_running_mean, module.project_conv_bn_running_var, 
                      module.project_conv_bn_weight, module.project_conv_bn_bias, training=module.training)
    
    if module.use_residual:
        x += identity
    
    return x

# Test code
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]
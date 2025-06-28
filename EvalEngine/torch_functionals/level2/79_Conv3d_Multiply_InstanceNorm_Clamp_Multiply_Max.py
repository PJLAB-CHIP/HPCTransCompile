import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    multiplier: torch.Tensor,
    clamp_min: float,
    clamp_max: float
) -> torch.Tensor:
    """
    3D卷积后接乘法、实例归一化、截断、再乘法和最大值操作的函数式实现
    
    Args:
        x: 输入张量 (B, C_in, D, H, W)
        conv_weight: 卷积核权重 (C_out, C_in, kD, kH, kW)
        conv_bias: 卷积偏置 (C_out,)
        multiplier: 乘法系数 (C_out, 1, 1, 1)
        clamp_min: 截断下限
        clamp_max: 截断上限
        
    Returns:
        处理后的张量 (B, 1, D', H', W')
    """
    x = F.conv3d(x, conv_weight, conv_bias)
    x = x * multiplier
    x = F.instance_norm(x, running_mean=None, running_var=None)
    x = torch.clamp(x, clamp_min, clamp_max)
    x = x * multiplier
    x = torch.max(x, dim=1, keepdim=True)[0]
    return x

class Model(nn.Module):
    """
    封装卷积参数并提供forward接口的模型类
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(Model, self).__init__()
        # 初始化卷积参数
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        
        # 初始化其他参数
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.multiplier,
            self.clamp_min,
            self.clamp_max
        )

# 参数配置与原始代码保持一致
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
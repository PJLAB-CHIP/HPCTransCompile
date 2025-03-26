import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    """
    This function ensures that the number of channels is divisible by the divisor.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def inverted_residual_block_fn(x, params, prefix, use_res_connect):
    """
    Functional version of the inverted residual block.
    """
    if prefix + 'conv1.weight' in params:
        # Pointwise convolution
        x = F.conv2d(x, params[prefix + 'conv1.weight'], stride=1, padding=0)
        x = F.batch_norm(x, params.get(prefix + 'bn1.weight'), params.get(prefix + 'bn1.bias'), 
                         params.get(prefix + 'bn1.running_mean'), params.get(prefix + 'bn1.running_var'), training=False)
        x = F.relu6(x, inplace=True)

    # Depthwise convolution
    x = F.conv2d(x, params[prefix + 'conv2.weight'], stride=params[prefix + 'stride'], padding=1, 
                 groups=x.size(1))
    x = F.batch_norm(x, params.get(prefix + 'bn2.weight'), params.get(prefix + 'bn2.bias'), 
                     params.get(prefix + 'bn2.running_mean'), params.get(prefix + 'bn2.running_var'), training=False)
    x = F.relu6(x, inplace=True)

    # Pointwise linear convolution
    x = F.conv2d(x, params[prefix + 'conv3.weight'], stride=1, padding=0)
    x = F.batch_norm(x, params.get(prefix + 'bn3.weight'), params.get(prefix + 'bn3.bias'), 
                     params.get(prefix + 'bn3.running_mean'), params.get(prefix + 'bn3.running_var'), training=False)

    if use_res_connect:
        return x + params[prefix + 'residual']
    else:
        return x

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Parameters for the first layer
        self.conv1_weight = nn.Parameter(torch.empty(32, 3, 3, 3))
        self.bn1_weight = nn.Parameter(torch.empty(32))
        self.bn1_bias = nn.Parameter(torch.empty(32))
        self.bn1_running_mean = nn.Parameter(torch.empty(32), requires_grad=False)
        self.bn1_running_var = nn.Parameter(torch.empty(32), requires_grad=False)

        # Parameters for inverted residual blocks
        self.ir_params = nn.ParameterDict()
        block_idx = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                hidden_dim = int(input_channel * t)
                use_res_connect = stride == 1 and input_channel == output_channel

                prefix = f'block{block_idx}.'
                if t != 1:
                    self.ir_params[prefix + 'conv1.weight'] = nn.Parameter(torch.empty(hidden_dim, input_channel, 1, 1))
                    self.ir_params[prefix + 'bn1.weight'] = nn.Parameter(torch.empty(hidden_dim))
                    self.ir_params[prefix + 'bn1.bias'] = nn.Parameter(torch.empty(hidden_dim))
                    self.ir_params[prefix + 'bn1.running_mean'] = nn.Parameter(torch.empty(hidden_dim), requires_grad=False)
                    self.ir_params[prefix + 'bn1.running_var'] = nn.Parameter(torch.empty(hidden_dim), requires_grad=False)

                self.ir_params[prefix + 'conv2.weight'] = nn.Parameter(torch.empty(hidden_dim, 1, 3, 3))
                self.ir_params[prefix + 'bn2.weight'] = nn.Parameter(torch.empty(hidden_dim))
                self.ir_params[prefix + 'bn2.bias'] = nn.Parameter(torch.empty(hidden_dim))
                self.ir_params[prefix + 'bn2.running_mean'] = nn.Parameter(torch.empty(hidden_dim), requires_grad=False)
                self.ir_params[prefix + 'bn2.running_var'] = nn.Parameter(torch.empty(hidden_dim), requires_grad=False)
                self.ir_params[prefix + 'conv3.weight'] = nn.Parameter(torch.empty(output_channel, hidden_dim, 1, 1))
                self.ir_params[prefix + 'bn3.weight'] = nn.Parameter(torch.empty(output_channel))
                self.ir_params[prefix + 'bn3.bias'] = nn.Parameter(torch.empty(output_channel))
                self.ir_params[prefix + 'bn3.running_mean'] = nn.Parameter(torch.empty(output_channel), requires_grad=False)
                self.ir_params[prefix + 'bn3.running_var'] = nn.Parameter(torch.empty(output_channel), requires_grad=False)
                self.ir_params[prefix + 'stride'] = nn.Parameter(torch.tensor(stride), requires_grad=False)
                if use_res_connect:
                    self.ir_params[prefix + 'residual'] = nn.Parameter(torch.zeros(1, output_channel, 1, 1), requires_grad=False)

                input_channel = output_channel
                block_idx += 1

        # Parameters for the last layers
        self.conv_last_weight = nn.Parameter(torch.empty(last_channel, input_channel, 1, 1))
        self.bn_last_weight = nn.Parameter(torch.empty(last_channel))
        self.bn_last_bias = nn.Parameter(torch.empty(last_channel))
        self.bn_last_running_mean = nn.Parameter(torch.empty(last_channel), requires_grad=False)
        self.bn_last_running_var = nn.Parameter(torch.empty(last_channel), requires_grad=False)

        # Parameters for the classifier
        self.fc_weight = nn.Parameter(torch.empty(num_classes, last_channel))
        self.fc_bias = nn.Parameter(torch.empty(num_classes))

        # Initialize weights
        nn.init.kaiming_normal_(self.conv1_weight, mode='fan_out')
        nn.init.ones_(self.bn1_weight)
        nn.init.zeros_(self.bn1_bias)
        nn.init.zeros_(self.bn1_running_mean)
        nn.init.ones_(self.bn1_running_var)

        for name, param in self.ir_params.items():
            if 'weight' in name and 'conv' in name:
                nn.init.kaiming_normal_(param, mode='fan_out')
            elif 'weight' in name and 'bn' in name:
                nn.init.ones_(param)
            elif 'bias' in name and 'bn' in name:
                nn.init.zeros_(param)
            elif 'running_mean' in name:
                nn.init.zeros_(param)
            elif 'running_var' in name:
                nn.init.ones_(param)

        nn.init.kaiming_normal_(self.conv_last_weight, mode='fan_out')
        nn.init.ones_(self.bn_last_weight)
        nn.init.zeros_(self.bn_last_bias)
        nn.init.zeros_(self.bn_last_running_mean)
        nn.init.ones_(self.bn_last_running_var)

        nn.init.normal_(self.fc_weight, 0, 0.01)
        nn.init.zeros_(self.fc_bias)

    def forward(self, x, fn=None):
        if fn is None:
            fn = module_fn
        return fn(x, self)

def module_fn(x, model):
    # First layer
    x = F.conv2d(x, model.conv1_weight, stride=2, padding=1)
    x = F.batch_norm(x, model.bn1_weight, model.bn1_bias, 
                     model.bn1_running_mean, model.bn1_running_var, training=False)
    x = F.relu6(x, inplace=True)

    # Inverted residual blocks
    block_idx = 0
    for name, param in model.ir_params.items():
        if name.startswith(f'block{block_idx}.'):
            prefix = f'block{block_idx}.'
            use_res_connect = prefix + 'residual' in model.ir_params
            x = inverted_residual_block_fn(x, model.ir_params, prefix, use_res_connect)
            block_idx += 1

    # Last layers
    x = F.conv2d(x, model.conv_last_weight, stride=1, padding=0)
    x = F.batch_norm(x, model.bn_last_weight, model.bn_last_bias, 
                     model.bn_last_running_mean, model.bn_last_running_var, training=False)
    x = F.relu6(x, inplace=True)

    # Adaptive average pooling
    x = F.adaptive_avg_pool2d(x, (1, 1))

    # Classifier
    x = x.view(x.size(0), -1)
    x = F.linear(x, model.fc_weight, model.fc_bias)
    return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
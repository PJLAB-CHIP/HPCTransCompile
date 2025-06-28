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
    if prefix + 'conv1_weight' in params:
        # Pointwise convolution
        x = F.conv2d(x, params[prefix + 'conv1_weight'], stride=1, padding=0)
        x = F.batch_norm(x, 
                         params.get(prefix + 'bn1_running_mean'), 
                         params.get(prefix + 'bn1_running_var'), 
                         params.get(prefix + 'bn1_weight'), 
                         params.get(prefix + 'bn1_bias'), 
                         training=False)
        x = F.relu6(x, inplace=True)

    # Depthwise convolution
    x = F.conv2d(x, params[prefix + 'conv2_weight'], stride=params[prefix + 'stride'].item(), padding=1, 
                 groups=x.size(1))
    x = F.batch_norm(x, 
                    params.get(prefix + 'bn2_running_mean'), 
                    params.get(prefix + 'bn2_running_var'), 
                    params.get(prefix + 'bn2_weight'), 
                    params.get(prefix + 'bn2_bias'), 
                    training=False)
    x = F.relu6(x, inplace=True)

    # Pointwise linear convolution
    x = F.conv2d(x, params[prefix + 'conv3_weight'], stride=1, padding=0)
    x = F.batch_norm(x, 
                    params.get(prefix + 'bn3_running_mean'), 
                    params.get(prefix + 'bn3_running_var'), 
                    params.get(prefix + 'bn3_weight'), 
                    params.get(prefix + 'bn3_bias'), 
                    training=False)

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
        self.params = nn.ParameterDict()

        # Parameters for the first layer
        self.params['conv0_weight'] = nn.Parameter(torch.empty(32, 3, 3, 3))
        self.params['bn0_weight'] = nn.Parameter(torch.empty(32))
        self.params['bn0_bias'] = nn.Parameter(torch.empty(32))
        self.params['bn0_running_mean'] = nn.Parameter(torch.empty(32), requires_grad=False)
        self.params['bn0_running_var'] = nn.Parameter(torch.empty(32), requires_grad=False)

        # Parameters for inverted residual blocks
        block_idx = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                hidden_dim = int(input_channel * t)
                use_res_connect = stride == 1 and input_channel == output_channel

                prefix = f'block{block_idx}_'
                if t != 1:
                    self.params[prefix + 'conv1_weight'] = nn.Parameter(torch.empty(hidden_dim, input_channel, 1, 1))
                    self.params[prefix + 'bn1_weight'] = nn.Parameter(torch.empty(hidden_dim))
                    self.params[prefix + 'bn1_bias'] = nn.Parameter(torch.empty(hidden_dim))
                    self.params[prefix + 'bn1_running_mean'] = nn.Parameter(torch.empty(hidden_dim), requires_grad=False)
                    self.params[prefix + 'bn1_running_var'] = nn.Parameter(torch.empty(hidden_dim), requires_grad=False)

                self.params[prefix + 'conv2_weight'] = nn.Parameter(torch.empty(hidden_dim, 1, 3, 3))
                self.params[prefix + 'bn2_weight'] = nn.Parameter(torch.empty(hidden_dim))
                self.params[prefix + 'bn2_bias'] = nn.Parameter(torch.empty(hidden_dim))
                self.params[prefix + 'bn2_running_mean'] = nn.Parameter(torch.empty(hidden_dim), requires_grad=False)
                self.params[prefix + 'bn2_running_var'] = nn.Parameter(torch.empty(hidden_dim), requires_grad=False)
                self.params[prefix + 'conv3_weight'] = nn.Parameter(torch.empty(output_channel, hidden_dim, 1, 1))
                self.params[prefix + 'bn3_weight'] = nn.Parameter(torch.empty(output_channel))
                self.params[prefix + 'bn3_bias'] = nn.Parameter(torch.empty(output_channel))
                self.params[prefix + 'bn3_running_mean'] = nn.Parameter(torch.empty(output_channel), requires_grad=False)
                self.params[prefix + 'bn3_running_var'] = nn.Parameter(torch.empty(output_channel), requires_grad=False)
                self.params[prefix + 'stride'] = nn.Parameter(torch.tensor(stride), requires_grad=False)
                if use_res_connect:
                    self.params[prefix + 'residual'] = nn.Parameter(torch.zeros(1, output_channel, 1, 1), requires_grad=False)

                input_channel = output_channel
                block_idx += 1

        # Parameters for the last layers
        self.params['conv_last_weight'] = nn.Parameter(torch.empty(last_channel, input_channel, 1, 1))
        self.params['bn_last_weight'] = nn.Parameter(torch.empty(last_channel))
        self.params['bn_last_bias'] = nn.Parameter(torch.empty(last_channel))
        self.params['bn_last_running_mean'] = nn.Parameter(torch.empty(last_channel), requires_grad=False)
        self.params['bn_last_running_var'] = nn.Parameter(torch.empty(last_channel), requires_grad=False)

        # Parameters for the classifier
        self.params['fc_weight'] = nn.Parameter(torch.empty(num_classes, last_channel))
        self.params['fc_bias'] = nn.Parameter(torch.empty(num_classes))

        # Initialize weights
        # nn.init.kaiming_normal_(self.conv1_weight, mode='fan_out')
        # nn.init.ones_(self.bn1_weight)
        # nn.init.zeros_(self.bn1_bias)
        # nn.init.zeros_(self.bn1_running_mean)
        # nn.init.ones_(self.bn1_running_var)

        for name, param in self.params.items():
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

        # nn.init.kaiming_normal_(self.conv_last_weight, mode='fan_out')
        # nn.init.ones_(self.bn_last_weight)
        # nn.init.zeros_(self.bn_last_bias)
        # nn.init.zeros_(self.bn_last_running_mean)
        # nn.init.ones_(self.bn_last_running_var)

        # nn.init.normal_(self.fc_weight, 0, 0.01)
        # nn.init.zeros_(self.fc_bias)

    def forward(self, x, fn=None):
        if fn is None:
            fn = module_fn
        return fn(x, self.params, False)

def module_fn(x, params, is_training):
    # First layer
    x = F.conv2d(x, params['conv0_weight'], stride=2, padding=1)
    x = F.batch_norm(x, 
                     params['bn0_running_mean'], 
                     params['bn0_running_var'], 
                     params['bn0_weight'], 
                     params['bn0_bias'], 
                     training=False)
    x = F.relu6(x, inplace=True)

    # Inverted residual blocks
    block_idx = 0
    for name, param in params.items():
        if name.startswith(f'block{block_idx}_'):
            prefix = f'block{block_idx}_'
            use_res_connect = prefix + 'residual' in params
            x = inverted_residual_block_fn(x, params, prefix, use_res_connect)
            
            block_idx += 1

    # Last layers
    x = F.conv2d(x, params['conv_last_weight'], stride=1, padding=0)
    x = F.batch_norm(x, 
                     params['bn_last_running_mean'], 
                     params['bn_last_running_var'], 
                     params['bn_last_weight'], 
                     params['bn_last_bias'], 
                     training=False)
    x = F.relu6(x, inplace=True)
    # Adaptive average pooling
    x = F.adaptive_avg_pool2d(x, (1, 1))

    # Classifier
    x = x.view(x.size(0), -1)
    x = F.linear(x, params['fc_weight'], params['fc_bias'])
    return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups
    
    # Reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    
    # Transpose
    x = x.transpose(1, 2).contiguous()
    
    # Flatten
    x = x.view(batch_size, -1, height, width)
    
    return x

def shuffle_net_unit_fn(x, conv1_weight, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, 
                        conv2_weight, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
                        conv3_weight, bn3_weight, bn3_bias, bn3_running_mean, bn3_running_var,
                        shortcut_conv_weight, shortcut_bn_weight, shortcut_bn_bias, 
                        shortcut_bn_running_mean, shortcut_bn_running_var, in_channels, out_channels, groups):
    # First 1x1 group convolution
    out = F.conv2d(x, conv1_weight, stride=1, padding=0, groups=groups)
    out = F.batch_norm(out, bn1_running_mean, bn1_running_var, bn1_weight, bn1_bias, training=False)
    out = F.relu(out)
    
    # Depthwise 3x3 convolution
    out = F.conv2d(out, conv2_weight, stride=1, padding=1, groups=out.size(1))
    out = F.batch_norm(out, bn2_running_mean, bn2_running_var, bn2_weight, bn2_bias, training=False)
    
    # Shuffle operation
    out = channel_shuffle(out, groups)
    
    # Second 1x1 group convolution
    out = F.conv2d(out, conv3_weight, stride=1, padding=0, groups=groups)
    out = F.batch_norm(out, bn3_running_mean, bn3_running_var, bn3_weight, bn3_bias, training=False)
    out = F.relu(out)
    
    # Shortcut connection
    if in_channels == out_channels:
        shortcut = x
    else:
        shortcut = F.conv2d(x, shortcut_conv_weight, stride=1, padding=0)
        shortcut = F.batch_norm(shortcut, shortcut_bn_running_mean, shortcut_bn_running_var, 
                                shortcut_bn_weight, shortcut_bn_bias, training=False)
    
    out += shortcut
    return out

class Model(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(Model, self).__init__()
        
        # conv1 and bn1
        self.conv1_weight = nn.Parameter(torch.empty(stages_out_channels[0], 3, 3, 3))
        self.bn1_weight = nn.Parameter(torch.empty(stages_out_channels[0]))
        self.bn1_bias = nn.Parameter(torch.empty(stages_out_channels[0]))
        self.register_buffer('bn1_running_mean', torch.zeros(stages_out_channels[0]))
        self.register_buffer('bn1_running_var', torch.ones(stages_out_channels[0]))
        
        # stage2, stage3, stage4
        self.stage2_params = self._make_stage_params(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3_params = self._make_stage_params(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4_params = self._make_stage_params(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        # conv5 and bn5
        self.conv5_weight = nn.Parameter(torch.empty(1024, stages_out_channels[3], 1, 1))
        self.bn5_weight = nn.Parameter(torch.empty(1024))
        self.bn5_bias = nn.Parameter(torch.empty(1024))
        self.register_buffer('bn5_running_mean', torch.zeros(1024))
        self.register_buffer('bn5_running_var', torch.ones(1024))
        
        # fc
        self.fc_weight = nn.Parameter(torch.empty(num_classes, 1024))
        self.fc_bias = nn.Parameter(torch.empty(num_classes))
        
        # Initialize parameters
        nn.init.kaiming_normal_(self.conv1_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.bn1_weight, 1)
        nn.init.constant_(self.bn1_bias, 0)
        nn.init.constant_(self.bn5_weight, 1)
        nn.init.constant_(self.bn5_bias, 0)
        nn.init.constant_(self.fc_bias, 0)
    
    def _make_stage_params(self, in_channels, out_channels, repeats, groups):
        params = nn.ParameterDict()
        for i in range(repeats):
            prefix = f'unit_{i}_'
            if i == 0:
                unit_in_channels = in_channels
            else:
                unit_in_channels = out_channels
            
            # conv1
            mid_channels = out_channels // 4
            params[prefix + 'conv1_weight'] = nn.Parameter(torch.empty(mid_channels, unit_in_channels // groups, 1, 1))
            params[prefix + 'bn1_weight'] = nn.Parameter(torch.empty(mid_channels))
            params[prefix + 'bn1_bias'] = nn.Parameter(torch.empty(mid_channels))
            params.register_buffer(prefix + 'bn1_running_mean', torch.zeros(mid_channels))
            params.register_buffer(prefix + 'bn1_running_var', torch.ones(mid_channels))
            
            # conv2
            params[prefix + 'conv2_weight'] = nn.Parameter(torch.empty(mid_channels, 1, 3, 3))
            params[prefix + 'bn2_weight'] = nn.Parameter(torch.empty(mid_channels))
            params[prefix + 'bn2_bias'] = nn.Parameter(torch.empty(mid_channels))
            params.register_buffer(prefix + 'bn2_running_mean', torch.zeros(mid_channels))
            params.register_buffer(prefix + 'bn2_running_var', torch.ones(mid_channels))
            
            # conv3
            params[prefix + 'conv3_weight'] = nn.Parameter(torch.empty(out_channels, mid_channels // groups, 1, 1))
            params[prefix + 'bn3_weight'] = nn.Parameter(torch.empty(out_channels))
            params[prefix + 'bn3_bias'] = nn.Parameter(torch.empty(out_channels))
            params.register_buffer(prefix + 'bn3_running_mean', torch.zeros(out_channels))
            params.register_buffer(prefix + 'bn3_running_var', torch.ones(out_channels))
            
            # shortcut
            if unit_in_channels != out_channels:
                params[prefix + 'shortcut_conv_weight'] = nn.Parameter(torch.empty(out_channels, unit_in_channels, 1, 1))
                params[prefix + 'shortcut_bn_weight'] = nn.Parameter(torch.empty(out_channels))
                params[prefix + 'shortcut_bn_bias'] = nn.Parameter(torch.empty(out_channels))
                params.register_buffer(prefix + 'shortcut_bn_running_mean', torch.zeros(out_channels))
                params.register_buffer(prefix + 'shortcut_bn_running_var', torch.ones(out_channels))
            
            # Initialize parameters
            nn.init.kaiming_normal_(params[prefix + 'conv1_weight'], mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(params[prefix + 'conv2_weight'], mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(params[prefix + 'conv3_weight'], mode='fan_out', nonlinearity='relu')
            if unit_in_channels != out_channels:
                nn.init.kaiming_normal_(params[prefix + 'shortcut_conv_weight'], mode='fan_out', nonlinearity='relu')
            nn.init.constant_(params[prefix + 'bn1_weight'], 1)
            nn.init.constant_(params[prefix + 'bn1_bias'], 0)
            nn.init.constant_(params[prefix + 'bn2_weight'], 1)
            nn.init.constant_(params[prefix + 'bn2_bias'], 0)
            nn.init.constant_(params[prefix + 'bn3_weight'], 1)
            nn.init.constant_(params[prefix + 'bn3_bias'], 0)
            if unit_in_channels != out_channels:
                nn.init.constant_(params[prefix + 'shortcut_bn_weight'], 1)
                nn.init.constant_(params[prefix + 'shortcut_bn_bias'], 0)
        
        return params
    
    def forward(self, x, fn=None):
        if fn is None:
            fn = self.module_fn
        return fn(x, **{name: param for name, param in self.named_parameters()}, 
                 **{name: buffer for name, buffer in self.named_buffers()})
    
    def module_fn(self, x, **params_and_buffers):
        # conv1 and bn1
        x = F.conv2d(x, self.conv1_weight, stride=2, padding=1)
        x = F.batch_norm(x, self.bn1_running_mean, self.bn1_running_var, self.bn1_weight, self.bn1_bias, training=False)
        x = F.relu(x)
        
        # maxpool
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        # stage2
        for i in range(len(self.stage2_params) // (6 if stages_out_channels[0] == stages_out_channels[1] else 8)):
            prefix = f'unit_{i}_'
            x = shuffle_net_unit_fn(
                x,
                self.stage2_params[prefix + 'conv1_weight'],
                self.stage2_params[prefix + 'bn1_weight'],
                self.stage2_params[prefix + 'bn1_bias'],
                self.stage2_params[prefix + 'bn1_running_mean'],
                self.stage2_params[prefix + 'bn1_running_var'],
                self.stage2_params[prefix + 'conv2_weight'],
                self.stage2_params[prefix + 'bn2_weight'],
                self.stage2_params[prefix + 'bn2_bias'],
                self.stage2_params[prefix + 'bn2_running_mean'],
                self.stage2_params[prefix + 'bn2_running_var'],
                self.stage2_params[prefix + 'conv3_weight'],
                self.stage2_params[prefix + 'bn3_weight'],
                self.stage2_params[prefix + 'bn3_bias'],
                self.stage2_params[prefix + 'bn3_running_mean'],
                self.stage2_params[prefix + 'bn3_running_var'],
                self.stage2_params.get(prefix + 'shortcut_conv_weight', None),
                self.stage2_params.get(prefix + 'shortcut_bn_weight', None),
                self.stage2_params.get(prefix + 'shortcut_bn_bias', None),
                self.stage2_params.get(prefix + 'shortcut_bn_running_mean', None),
                self.stage2_params.get(prefix + 'shortcut_bn_running_var', None),
                stages_out_channels[0] if i == 0 else stages_out_channels[1],
                stages_out_channels[1],
                groups
            )
        
        # stage3
        for i in range(len(self.stage3_params) // (6 if stages_out_channels[1] == stages_out_channels[2] else 8)):
            prefix = f'unit_{i}_'
            x = shuffle_net_unit_fn(
                x,
                self.stage3_params[prefix + 'conv1_weight'],
                self.stage3_params[prefix + 'bn1_weight'],
                self.stage3_params[prefix + 'bn1_bias'],
                self.stage3_params[prefix + 'bn1_running_mean'],
                self.stage3_params[prefix + 'bn1_running_var'],
                self.stage3_params[prefix + 'conv2_weight'],
                self.stage3_params[prefix + 'bn2_weight'],
                self.stage3_params[prefix + 'bn2_bias'],
                self.stage3_params[prefix + 'bn2_running_mean'],
                self.stage3_params[prefix + 'bn2_running_var'],
                self.stage3_params[prefix + 'conv3_weight'],
                self.stage3_params[prefix + 'bn3_weight'],
                self.stage3_params[prefix + 'bn3_bias'],
                self.stage3_params[prefix + 'bn3_running_mean'],
                self.stage3_params[prefix + 'bn3_running_var'],
                self.stage3_params.get(prefix + 'shortcut_conv_weight', None),
                self.stage3_params.get(prefix + 'shortcut_bn_weight', None),
                self.stage3_params.get(prefix + 'shortcut_bn_bias', None),
                self.stage3_params.get(prefix + 'shortcut_bn_running_mean', None),
                self.stage3_params.get(prefix + 'shortcut_bn_running_var', None),
                stages_out_channels[1] if i == 0 else stages_out_channels[2],
                stages_out_channels[2],
                groups
            )
        
        # stage4
        for i in range(len(self.stage4_params) // (6 if stages_out_channels[2] == stages_out_channels[3] else 8)):
            prefix = f'unit_{i}_'
            x = shuffle_net_unit_fn(
                x,
                self.stage4_params[prefix + 'conv1_weight'],
                self.stage4_params[prefix + 'bn1_weight'],
                self.stage4_params[prefix + 'bn1_bias'],
                self.stage4_params[prefix + 'bn1_running_mean'],
                self.stage4_params[prefix + 'bn1_running_var'],
                self.stage4_params[prefix + 'conv2_weight'],
                self.stage4_params[prefix + 'bn2_weight'],
                self.stage4_params[prefix + 'bn2_bias'],
                self.stage4_params[prefix + 'bn2_running_mean'],
                self.stage4_params[prefix + 'bn2_running_var'],
                self.stage4_params[prefix + 'conv3_weight'],
                self.stage4_params[prefix + 'bn3_weight'],
                self.stage4_params[prefix + 'bn3_bias'],
                self.stage4_params[prefix + 'bn3_running_mean'],
                self.stage4_params[prefix + 'bn3_running_var'],
                self.stage4_params.get(prefix + 'shortcut_conv_weight', None),
                self.stage4_params.get(prefix + 'shortcut_bn_weight', None),
                self.stage4_params.get(prefix + 'shortcut_bn_bias', None),
                self.stage4_params.get(prefix + 'shortcut_bn_running_mean', None),
                self.stage4_params.get(prefix + 'shortcut_bn_running_var', None),
                stages_out_channels[2] if i == 0 else stages_out_channels[3],
                stages_out_channels[3],
                groups
            )
        
        # conv5 and bn5
        x = F.conv2d(x, self.conv5_weight, stride=1, padding=0)
        x = F.batch_norm(x, self.bn5_running_mean, self.bn5_running_var, self.bn5_weight, self.bn5_bias, training=False)
        x = F.relu(x)
        
        # avg pool and fc
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = F.linear(x, self.fc_weight, self.fc_bias)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes] 
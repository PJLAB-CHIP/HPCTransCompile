import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

def shuffle_net_unit_fn(x, conv1_weight, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, 
                        conv2_weight, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
                        conv3_weight, bn3_weight, bn3_bias, bn3_running_mean, bn3_running_var,
                        shortcut_conv_weight, shortcut_bn_weight, shortcut_bn_bias, 
                        shortcut_bn_running_mean, shortcut_bn_running_var, in_channels, out_channels, groups):
    print("Input (first 5):", x[0, 0, 0, :5])
    out = F.conv2d(x, conv1_weight, stride=1, padding=0, groups=groups)
    print("After conv1 (first 5):", out[0, 0, 0, :5])
    out = F.batch_norm(out, bn1_running_mean, bn1_running_var, bn1_weight, bn1_bias, training=False)
    out = F.relu(out)
    print("After bn1+relu (first 5):", out[0, 0, 0, :5])
    out = F.conv2d(out, conv2_weight, stride=1, padding=1, groups=out.size(1))
    print("After conv2 (first 5):", out[0, 0, 0, :5])
    out = F.batch_norm(out, bn2_running_mean, bn2_running_var, bn2_weight, bn2_bias, training=False)
    print("After bn2 (first 5):", out[0, 0, 0, :5])
    out = channel_shuffle(out, groups)
    print("After channel_shuffle (first 5):", out[0, 0, 0, :5])
    out = F.conv2d(out, conv3_weight, stride=1, padding=0, groups=groups)
    print("After conv3 (first 5):", out[0, 0, 0, :5])
    out = F.batch_norm(out, bn3_running_mean, bn3_running_var, bn3_weight, bn3_bias, training=False)
    out = F.relu(out)
    print("After bn3+relu (first 5):", out[0, 0, 0, :5])
    if in_channels == out_channels:
        shortcut = x
    else:
        print("Shortcut conv weight (first 5):", shortcut_conv_weight.flatten()[:5])
        shortcut = F.conv2d(x, shortcut_conv_weight, stride=1, padding=0)
        print("Shortcut after conv (first 5):", shortcut[0, 0, 0, :5])
        if shortcut_bn_running_mean is not None and shortcut_bn_running_var is not None:
            shortcut = F.batch_norm(shortcut, shortcut_bn_running_mean, shortcut_bn_running_var, 
                                    shortcut_bn_weight, shortcut_bn_bias, training=False)
            print("Shortcut after bn (first 5):", shortcut[0, 0, 0, :5])
    out += shortcut
    return out

class Model(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(Model, self).__init__()
        self.stages_out_channels = stages_out_channels
        self.groups = groups
        self.stages_repeats = stages_repeats
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
        
        # Detach parameters to convert nn.Parameter to torch.Tensor, keep buffers as-is
        params = {name: param.detach() for name, param in self.named_parameters()}
        buffers = {name: buffer for name, buffer in self.named_buffers()}
        
        # Helper function to get tensor or zero tensor if None
        def get_tensor_or_zeros(key, shape):
            tensor = params.get(key, buffers.get(key))
            if tensor is not None:
                expected_elements = torch.prod(torch.tensor(shape)).item()
                if tensor.numel() != expected_elements:
                    print(f"Warning: {key} shape mismatch, expected {shape} ({expected_elements} elements), got {tensor.shape} ({tensor.numel()} elements)")
                    if tensor.numel() == expected_elements:
                        return tensor.view(shape)
                    else:
                        return torch.zeros(shape, device=x.device)
                return tensor
            return torch.zeros(shape, device=x.device)
                
        # 构造 stage 参数
        stage2_params = []
        for i in range(self.stages_repeats[0]):
            prefix = f'stage2_params.unit_{i}_'
            unit_in_channels = self.stages_out_channels[0] if i == 0 else self.stages_out_channels[1]
            out_channels = self.stages_out_channels[1]
            mid_channels = out_channels // 4
            stage2_params.extend([
                get_tensor_or_zeros(prefix + 'conv1_weight', (mid_channels, unit_in_channels // self.groups, 1, 1)),
                get_tensor_or_zeros(prefix + 'bn1_weight', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn1_bias', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn1_running_mean', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn1_running_var', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'conv2_weight', (mid_channels, 1, 3, 3)),
                get_tensor_or_zeros(prefix + 'bn2_weight', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn2_bias', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn2_running_mean', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn2_running_var', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'conv3_weight', (out_channels, mid_channels // self.groups, 1, 1)),
                get_tensor_or_zeros(prefix + 'bn3_weight', (out_channels,)),
                get_tensor_or_zeros(prefix + 'bn3_bias', (out_channels,)),
                get_tensor_or_zeros(prefix + 'bn3_running_mean', (out_channels,)),
                get_tensor_or_zeros(prefix + 'bn3_running_var', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_conv_weight', (out_channels, unit_in_channels, 1, 1)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_weight', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_bias', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_running_mean', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_running_var', (out_channels,))
            ])
        
        stage3_params = []
        for i in range(self.stages_repeats[1]):
            prefix = f'stage3_params.unit_{i}_'
            unit_in_channels = self.stages_out_channels[1] if i == 0 else self.stages_out_channels[2]
            out_channels = self.stages_out_channels[2]
            mid_channels = out_channels // 4
            stage3_params.extend([
                get_tensor_or_zeros(prefix + 'conv1_weight', (mid_channels, unit_in_channels // self.groups, 1, 1)),
                get_tensor_or_zeros(prefix + 'bn1_weight', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn1_bias', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn1_running_mean', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn1_running_var', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'conv2_weight', (mid_channels, 1, 3, 3)),
                get_tensor_or_zeros(prefix + 'bn2_weight', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn2_bias', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn2_running_mean', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn2_running_var', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'conv3_weight', (out_channels, mid_channels // self.groups, 1, 1)),
                get_tensor_or_zeros(prefix + 'bn3_weight', (out_channels,)),
                get_tensor_or_zeros(prefix + 'bn3_bias', (out_channels,)),
                get_tensor_or_zeros(prefix + 'bn3_running_mean', (out_channels,)),
                get_tensor_or_zeros(prefix + 'bn3_running_var', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_conv_weight', (out_channels, unit_in_channels, 1, 1)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_weight', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_bias', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_running_mean', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_running_var', (out_channels,))
            ])
        
        stage4_params = []
        for i in range(self.stages_repeats[2]):
            prefix = f'stage4_params.unit_{i}_'
            unit_in_channels = self.stages_out_channels[2] if i == 0 else self.stages_out_channels[3]
            out_channels = self.stages_out_channels[3]
            mid_channels = out_channels // 4
            stage4_params.extend([
                get_tensor_or_zeros(prefix + 'conv1_weight', (mid_channels, unit_in_channels // self.groups, 1, 1)),
                get_tensor_or_zeros(prefix + 'bn1_weight', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn1_bias', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn1_running_mean', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn1_running_var', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'conv2_weight', (mid_channels, 1, 3, 3)),
                get_tensor_or_zeros(prefix + 'bn2_weight', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn2_bias', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn2_running_mean', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'bn2_running_var', (mid_channels,)),
                get_tensor_or_zeros(prefix + 'conv3_weight', (out_channels, mid_channels // self.groups, 1, 1)),
                get_tensor_or_zeros(prefix + 'bn3_weight', (out_channels,)),
                get_tensor_or_zeros(prefix + 'bn3_bias', (out_channels,)),
                get_tensor_or_zeros(prefix + 'bn3_running_mean', (out_channels,)),
                get_tensor_or_zeros(prefix + 'bn3_running_var', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_conv_weight', (out_channels, unit_in_channels, 1, 1)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_weight', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_bias', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_running_mean', (out_channels,)),
                get_tensor_or_zeros(prefix + 'shortcut_bn_running_var', (out_channels,))
            ])

        # 根据 fn 类型选择调用方式
        if fn == self.module_fn:
            # 为 module_fn 准备字典形式的参数
            params_and_buffers = {
                'conv1_weight': params['conv1_weight'],
                'bn1_weight': params['bn1_weight'],
                'bn1_bias': params['bn1_bias'],
                'bn1_running_mean': buffers['bn1_running_mean'],
                'bn1_running_var': buffers['bn1_running_var'],
                'conv5_weight': params['conv5_weight'],
                'bn5_weight': params['bn5_weight'],
                'bn5_bias': params['bn5_bias'],
                'bn5_running_mean': buffers['bn5_running_mean'],
                'bn5_running_var': buffers['bn5_running_var'],
                'fc_weight': params['fc_weight'],
                'fc_bias': params['fc_bias'],
            }
            # 将 stage 参数转为字典形式
            stage2_params_dict = {}
            for i in range(self.stages_repeats[0]):
                prefix = f'unit_{i}_'
                base_idx = i * 20
                stage2_params_dict[prefix + 'conv1_weight'] = stage2_params[base_idx]
                stage2_params_dict[prefix + 'bn1_weight'] = stage2_params[base_idx + 1]
                stage2_params_dict[prefix + 'bn1_bias'] = stage2_params[base_idx + 2]
                stage2_params_dict[prefix + 'bn1_running_mean'] = stage2_params[base_idx + 3]
                stage2_params_dict[prefix + 'bn1_running_var'] = stage2_params[base_idx + 4]
                stage2_params_dict[prefix + 'conv2_weight'] = stage2_params[base_idx + 5]
                stage2_params_dict[prefix + 'bn2_weight'] = stage2_params[base_idx + 6]
                stage2_params_dict[prefix + 'bn2_bias'] = stage2_params[base_idx + 7]
                stage2_params_dict[prefix + 'bn2_running_mean'] = stage2_params[base_idx + 8]
                stage2_params_dict[prefix + 'bn2_running_var'] = stage2_params[base_idx + 9]
                stage2_params_dict[prefix + 'conv3_weight'] = stage2_params[base_idx + 10]
                stage2_params_dict[prefix + 'bn3_weight'] = stage2_params[base_idx + 11]
                stage2_params_dict[prefix + 'bn3_bias'] = stage2_params[base_idx + 12]
                stage2_params_dict[prefix + 'bn3_running_mean'] = stage2_params[base_idx + 13]
                stage2_params_dict[prefix + 'bn3_running_var'] = stage2_params[base_idx + 14]
                stage2_params_dict[prefix + 'shortcut_conv_weight'] = stage2_params[base_idx + 15]
                stage2_params_dict[prefix + 'shortcut_bn_weight'] = stage2_params[base_idx + 16]
                stage2_params_dict[prefix + 'shortcut_bn_bias'] = stage2_params[base_idx + 17]
                stage2_params_dict[prefix + 'shortcut_bn_running_mean'] = stage2_params[base_idx + 18]
                stage2_params_dict[prefix + 'shortcut_bn_running_var'] = stage2_params[base_idx + 19]
            
            stage3_params_dict = {}
            for i in range(self.stages_repeats[1]):
                prefix = f'unit_{i}_'
                base_idx = i * 20
                stage3_params_dict[prefix + 'conv1_weight'] = stage3_params[base_idx]
                stage3_params_dict[prefix + 'bn1_weight'] = stage3_params[base_idx + 1]
                stage3_params_dict[prefix + 'bn1_bias'] = stage3_params[base_idx + 2]
                stage3_params_dict[prefix + 'bn1_running_mean'] = stage3_params[base_idx + 3]
                stage3_params_dict[prefix + 'bn1_running_var'] = stage3_params[base_idx + 4]
                stage3_params_dict[prefix + 'conv2_weight'] = stage3_params[base_idx + 5]
                stage3_params_dict[prefix + 'bn2_weight'] = stage3_params[base_idx + 6]
                stage3_params_dict[prefix + 'bn2_bias'] = stage3_params[base_idx + 7]
                stage3_params_dict[prefix + 'bn2_running_mean'] = stage3_params[base_idx + 8]
                stage3_params_dict[prefix + 'bn2_running_var'] = stage3_params[base_idx + 9]
                stage3_params_dict[prefix + 'conv3_weight'] = stage3_params[base_idx + 10]
                stage3_params_dict[prefix + 'bn3_weight'] = stage3_params[base_idx + 11]
                stage3_params_dict[prefix + 'bn3_bias'] = stage3_params[base_idx + 12]
                stage3_params_dict[prefix + 'bn3_running_mean'] = stage3_params[base_idx + 13]
                stage3_params_dict[prefix + 'bn3_running_var'] = stage3_params[base_idx + 14]
                stage3_params_dict[prefix + 'shortcut_conv_weight'] = stage3_params[base_idx + 15]
                stage3_params_dict[prefix + 'shortcut_bn_weight'] = stage3_params[base_idx + 16]
                stage3_params_dict[prefix + 'shortcut_bn_bias'] = stage3_params[base_idx + 17]
                stage3_params_dict[prefix + 'shortcut_bn_running_mean'] = stage3_params[base_idx + 18]
                stage3_params_dict[prefix + 'shortcut_bn_running_var'] = stage3_params[base_idx + 19]
            
            stage4_params_dict = {}
            for i in range(self.stages_repeats[2]):
                prefix = f'unit_{i}_'
                base_idx = i * 20
                stage4_params_dict[prefix + 'conv1_weight'] = stage4_params[base_idx]
                stage4_params_dict[prefix + 'bn1_weight'] = stage4_params[base_idx + 1]
                stage4_params_dict[prefix + 'bn1_bias'] = stage4_params[base_idx + 2]
                stage4_params_dict[prefix + 'bn1_running_mean'] = stage4_params[base_idx + 3]
                stage4_params_dict[prefix + 'bn1_running_var'] = stage4_params[base_idx + 4]
                stage4_params_dict[prefix + 'conv2_weight'] = stage4_params[base_idx + 5]
                stage4_params_dict[prefix + 'bn2_weight'] = stage4_params[base_idx + 6]
                stage4_params_dict[prefix + 'bn2_bias'] = stage4_params[base_idx + 7]
                stage4_params_dict[prefix + 'bn2_running_mean'] = stage4_params[base_idx + 8]
                stage4_params_dict[prefix + 'bn2_running_var'] = stage4_params[base_idx + 9]
                stage4_params_dict[prefix + 'conv3_weight'] = stage4_params[base_idx + 10]
                stage4_params_dict[prefix + 'bn3_weight'] = stage4_params[base_idx + 11]
                stage4_params_dict[prefix + 'bn3_bias'] = stage4_params[base_idx + 12]
                stage4_params_dict[prefix + 'bn3_running_mean'] = stage4_params[base_idx + 13]
                stage4_params_dict[prefix + 'bn3_running_var'] = stage4_params[base_idx + 14]
                stage4_params_dict[prefix + 'shortcut_conv_weight'] = stage4_params[base_idx + 15]
                stage4_params_dict[prefix + 'shortcut_bn_weight'] = stage4_params[base_idx + 16]
                stage4_params_dict[prefix + 'shortcut_bn_bias'] = stage4_params[base_idx + 17]
                stage4_params_dict[prefix + 'shortcut_bn_running_mean'] = stage4_params[base_idx + 18]
                stage4_params_dict[prefix + 'shortcut_bn_running_var'] = stage4_params[base_idx + 19]
            
            params_and_buffers['stage2_params'] = stage2_params_dict
            params_and_buffers['stage3_params'] = stage3_params_dict
            params_and_buffers['stage4_params'] = stage4_params_dict
            return fn(x, **params_and_buffers)
        else:
            # 假设 fn 是 CUDA forward，使用位置参数
            return fn(
                x,
                params['conv1_weight'],
                params['bn1_weight'],
                params['bn1_bias'],
                buffers['bn1_running_mean'],
                buffers['bn1_running_var'],
                params['conv5_weight'],
                params['bn5_weight'],
                params['bn5_bias'],
                buffers['bn5_running_mean'],
                buffers['bn5_running_var'],
                params['fc_weight'],
                params['fc_bias'],
                stage2_params,
                stage3_params,
                stage4_params,
                self.stages_out_channels[0],  # in_channels
                self.groups,
                self.stages_repeats[0],       # stage2_repeats
                self.stages_repeats[1],       # stage3_repeats
                self.stages_repeats[2],       # stage4_repeats
                self.stages_out_channels
            )
            
    def module_fn(self, x, **params_and_buffers):
        # conv1 and bn1
        x = F.conv2d(x, self.conv1_weight, stride=2, padding=1)
        print("Python conv1 output: ", x.shape, x[0])
        x = F.batch_norm(x, self.bn1_running_mean, self.bn1_running_var, self.bn1_weight, self.bn1_bias, training=False)
        print("Python bn1 output: ", x.shape, x[0])
        return x

    def module_fn(self, x, **params_and_buffers):
        # conv1 and bn1
        x = F.conv2d(x, self.conv1_weight, stride=2, padding=1)
        x = F.batch_norm(x, self.bn1_running_mean, self.bn1_running_var, self.bn1_weight, self.bn1_bias, training=False)
        x = F.relu(x)
        
        # maxpool
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        # stage2
        for i in range(self.stages_repeats[0]):
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
                self.stages_out_channels[0] if i == 0 else self.stages_out_channels[1],
                self.stages_out_channels[1],
                self.groups
            )
        
        # stage3
        for i in range(self.stages_repeats[1]):
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
                self.stages_out_channels[1] if i == 0 else self.stages_out_channels[2],
                self.stages_out_channels[2],
                self.groups
            )
        
        # stage4
        for i in range(self.stages_repeats[2]):
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
                self.stages_out_channels[2] if i == 0 else self.stages_out_channels[3],
                self.stages_out_channels[3],
                self.groups
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
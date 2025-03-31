import torch
import torch.nn as nn
import torch.nn.functional as F

def transformer_encoder_layer_fn(x, self_attn, linear1, linear2, norm1, norm2, num_heads, dropout=0.0):
    """
    Functional version of nn.TransformerEncoderLayer.
    """
    # Self-attention part
    attn_output = self_attn(x)
    print(attn_output.size())
    x = x + attn_output
    x = norm1(x)

    # Feedforward part
    ff_output = linear2(F.gelu(linear1(x)))
    x = x + ff_output
    x = norm2(x)

    return x

def module_fn(x, params,
              num_heads, num_layers=6, patch_size=4, embed_dim=512, mlp_ratio=4.0):
    B, C, H, W = x.shape
    conv1_weight = params["conv1_weight"]
    conv1_bias = params["conv1_bias"]
    linear_proj_weight = params["linear_proj_weight"]
    linear_proj_bias = params["linear_proj_bias"]
    cls_token = params["cls_token"]
    transformer_layers_self_attn_in_proj_weight = params["transformer_layers_self_attn_in_proj_weight"]
    transformer_layers_self_attn_in_proj_bias = params["transformer_layers_self_attn_in_proj_bias"]
    transformer_layers_self_attn_out_proj_weight = params["transformer_layers_self_attn_out_proj_weight"]
    transformer_layers_self_attn_out_proj_bias = params["transformer_layers_self_attn_out_proj_bias"]
    transformer_layers_linear1_weight = params["transformer_layers_linear1_weight"]
    transformer_layers_linear1_bias = params["transformer_layers_linear1_bias"]
    transformer_layers_linear2_weight = params["transformer_layers_linear2_weight"]
    transformer_layers_linear2_bias = params["transformer_layers_linear2_bias"]
    transformer_layers_norm1_weight = params["transformer_layers_norm1_weight"]
    transformer_layers_norm1_bias = params["transformer_layers_norm1_bias"]
    transformer_layers_norm2_weight = params["transformer_layers_norm2_weight"]
    transformer_layers_norm2_bias = params["transformer_layers_norm2_bias"]

    transformer_layers_norm2_weight = params["transformer_layers_norm2_weight"]
    transformer_layers_norm2_weight = params["transformer_layers_norm2_weight"]
    transformer_layers_norm2_weight = params["transformer_layers_norm2_weight"]
    fc_out_weight = params["fc_out_weight"] 
    fc_out_bias = params["fc_out_bias"]

    # Convolutional patch embedding
    x = F.conv2d(x, conv1_weight, conv1_bias, stride=patch_size)
    x = x.flatten(1)  # (B, embed_dim * (H/patch_size) * (W/patch_size))
    
    # Linear projection
    x = F.linear(x, linear_proj_weight, linear_proj_bias)  # (B, embed_dim)
    
    # Add cls token
    cls_tokens = cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
    x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 1+N, embed_dim)

    # Transformer layers
    for i in range(1):
        # Extract parameters for the current layer
        start_idx = i * embed_dim * 3
        end_idx = (i + 1) * embed_dim * 3
        in_proj_weight = transformer_layers_self_attn_in_proj_weight[i]
        in_proj_bias = transformer_layers_self_attn_in_proj_bias[i]
        out_proj_weight = transformer_layers_self_attn_out_proj_weight[i]
        out_proj_bias = transformer_layers_self_attn_out_proj_bias[i]
        linear1_weight = transformer_layers_linear1_weight[i]
        linear1_bias = transformer_layers_linear1_bias[i]
        linear2_weight = transformer_layers_linear2_weight[i]
        linear2_bias = transformer_layers_linear2_bias[i]
        norm1_weight = transformer_layers_norm1_weight[i]
        norm1_bias = transformer_layers_norm1_bias[i]
        norm2_weight = transformer_layers_norm2_weight[i]
        norm2_bias = transformer_layers_norm2_bias[i]

        # Define self-attention and linear layers as functions
        def self_attn(x):
            q, k, v = F.linear(x, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            return F.linear(attn_output, out_proj_weight, out_proj_bias)

        def linear1(x):
            return F.linear(x, linear1_weight, linear1_bias)

        def linear2(x):
            return F.linear(x, linear2_weight, linear2_bias)

        def norm1(x):
            return F.layer_norm(x, (embed_dim,), norm1_weight, norm1_bias)

        def norm2(x):
            return F.layer_norm(x, (embed_dim,), norm2_weight, norm2_bias)

        x = transformer_encoder_layer_fn(x, self_attn, linear1, linear2, norm1, norm2, num_heads)
    # Classify based on cls token
    x = x[:, 0]  # Get the cls token's output
    x = F.linear(x, fc_out_weight, fc_out_bias)  # (B, num_classes)
    
    return x

class Model(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3):
        super(Model, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio

        # Parameters for conv1
        self.conv1_weight = nn.Parameter(torch.empty(embed_dim, in_channels, patch_size, patch_size))
        self.conv1_bias = nn.Parameter(torch.empty(embed_dim))

        # Parameters for linear_proj
        self.linear_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim * (32 // patch_size) * (32 // patch_size)))
        self.linear_proj_bias = nn.Parameter(torch.empty(embed_dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Parameters for transformer layers
        # Self-attention in_proj and out_proj weights/biases
        self.transformer_layers_self_attn_in_proj_weight = nn.Parameter(
            torch.empty(num_layers, 3 * embed_dim, embed_dim)
        )
        self.transformer_layers_self_attn_in_proj_bias = nn.Parameter(
            torch.empty(num_layers, 3 * embed_dim)
        )
        self.transformer_layers_self_attn_out_proj_weight = nn.Parameter(
            torch.empty(num_layers, embed_dim, embed_dim)
        )
        self.transformer_layers_self_attn_out_proj_bias = nn.Parameter(
            torch.empty(num_layers, embed_dim)
        )

        # MLP weights/biases
        self.transformer_layers_linear1_weight = nn.Parameter(
            torch.empty(num_layers, int(embed_dim * mlp_ratio), embed_dim)
        )
        self.transformer_layers_linear1_bias = nn.Parameter(
            torch.empty(num_layers, int(embed_dim * mlp_ratio))
        )
        self.transformer_layers_linear2_weight = nn.Parameter(
            torch.empty(num_layers, embed_dim, int(embed_dim * mlp_ratio))
        )
        self.transformer_layers_linear2_bias = nn.Parameter(
            torch.empty(num_layers, embed_dim)
        )

        # Layer norm weights/biases
        self.transformer_layers_norm1_weight = nn.Parameter(
            torch.empty(num_layers, embed_dim)
        )
        self.transformer_layers_norm1_bias = nn.Parameter(
            torch.empty(num_layers, embed_dim)
        )
        self.transformer_layers_norm2_weight = nn.Parameter(
            torch.empty(num_layers, embed_dim)
        )
        self.transformer_layers_norm2_bias = nn.Parameter(
            torch.empty(num_layers, embed_dim)
        )

        # Output layer
        self.fc_out_weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        self.fc_out_bias = nn.Parameter(torch.empty(num_classes))

        # Initialize parameters
        self._reset_parameters()
        self.params = nn.ParameterDict()
        self.params["conv1_weight"] = self.conv1_weight
        self.params["conv1_bias"] = self.conv1_bias
        self.params["linear_proj_weight"] = self.linear_proj_weight

        self.params["linear_proj_bias"] = self.linear_proj_bias
        self.params["cls_token"] = self.cls_token
        self.params["transformer_layers_self_attn_in_proj_weight"] = self.transformer_layers_self_attn_in_proj_weight
        self.params["transformer_layers_self_attn_in_proj_bias"] = self.transformer_layers_self_attn_in_proj_bias
        self.params["transformer_layers_self_attn_out_proj_weight"] = self.transformer_layers_self_attn_out_proj_weight
        self.params["transformer_layers_self_attn_out_proj_bias"] = self.transformer_layers_self_attn_out_proj_bias
        self.params["transformer_layers_linear1_weight"] = self.transformer_layers_linear1_weight
        self.params["transformer_layers_linear1_bias"] = self.transformer_layers_linear1_bias
        self.params["transformer_layers_linear2_weight"] = self.transformer_layers_linear2_weight
        self.params["transformer_layers_linear2_bias"] = self.transformer_layers_linear2_bias
        self.params["transformer_layers_norm1_weight"] = self.transformer_layers_norm1_weight
        self.params["transformer_layers_norm1_bias"] = self.transformer_layers_norm1_bias
        self.params["transformer_layers_norm2_weight"] = self.transformer_layers_norm2_weight
        self.params["transformer_layers_norm2_bias"] = self.transformer_layers_norm2_bias
        self.params["fc_out_weight"] = self.fc_out_weight
        self.params["fc_out_bias"] = self.fc_out_bias

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv1_weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.conv1_bias)
        nn.init.kaiming_uniform_(self.linear_proj_weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.linear_proj_bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for i in range(self.num_layers):
            nn.init.xavier_uniform_(self.transformer_layers_self_attn_in_proj_weight[i])
            nn.init.zeros_(self.transformer_layers_self_attn_in_proj_bias[i])
            nn.init.xavier_uniform_(self.transformer_layers_self_attn_out_proj_weight[i])
            nn.init.zeros_(self.transformer_layers_self_attn_out_proj_bias[i])
            nn.init.kaiming_uniform_(self.transformer_layers_linear1_weight[i], mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.transformer_layers_linear1_bias[i])
            nn.init.kaiming_uniform_(self.transformer_layers_linear2_weight[i], mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(self.transformer_layers_linear2_bias[i])
            nn.init.ones_(self.transformer_layers_norm1_weight[i])
            nn.init.zeros_(self.transformer_layers_norm1_bias[i])
            nn.init.ones_(self.transformer_layers_norm2_weight[i])
            nn.init.zeros_(self.transformer_layers_norm2_bias[i])
        nn.init.kaiming_uniform_(self.fc_out_weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.fc_out_bias)

    def forward(self, x, fn=module_fn):
        return fn(
            x, 
            self.params,
            self.num_heads, self.num_layers, self.patch_size, self.embed_dim, self.mlp_ratio
        )

batch_size = 10
image_size = 32
embed_dim = 128
in_channels = 3
num_heads = 4
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, in_channels, image_size, image_size)]

def get_init_inputs():
    return [num_classes, embed_dim, num_heads]
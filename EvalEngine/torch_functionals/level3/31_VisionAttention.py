import torch
import torch.nn as nn
import torch.nn.functional as F

def multihead_attention_forward(query, key, value, embed_dim, num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):
    # Reshape and split the weights and biases for Q, K, V
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    q = q * scaling
    
    q = q.contiguous().view(-1, q.size(1), num_heads, head_dim).transpose(1, 2)
    k = k.contiguous().view(-1, k.size(1), num_heads, head_dim).transpose(1, 2)
    v = v.contiguous().view(-1, v.size(1), num_heads, head_dim).transpose(1, 2)
    
    attn_output_weights = torch.matmul(q, k.transpose(-2, -1))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    
    attn_output = torch.matmul(attn_output_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, embed_dim)
    
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    return attn_output, attn_output_weights

def layer_norm(x, normalized_shape, weight, bias, eps=1e-5):
    return F.layer_norm(x, normalized_shape, weight, bias, eps)

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Parameters for MultiheadAttention
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.out_proj_bias = nn.Parameter(torch.empty(embed_dim))
        
        # Parameters for LayerNorm
        self.norm_weight = nn.Parameter(torch.ones(embed_dim))
        self.norm_bias = nn.Parameter(torch.zeros(embed_dim))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)
        nn.init.xavier_uniform_(self.out_proj_weight)
        nn.init.ones_(self.norm_weight)
        nn.init.zeros_(self.norm_bias)

    def forward(self, x, fn=None):
        if fn is None:
            fn = module_fn
        return fn(x, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.norm_weight, self.norm_bias)

def module_fn(x, embed_dim, num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, norm_weight, norm_bias):
    B, C, H, W = x.shape
    x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
    
    attn_output, _ = multihead_attention_forward(
        x, x, x, embed_dim, num_heads,
        in_proj_weight, in_proj_bias,
        out_proj_weight, out_proj_bias
    )
    
    x = layer_norm(attn_output + x, (embed_dim,), norm_weight, norm_bias)
    x = x.permute(1, 2, 0).view(B, C, H, W)
    return x

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.randn(batch_size, num_channels, image_height, image_width)]

def get_init_inputs():
    return [embed_dim, num_heads]
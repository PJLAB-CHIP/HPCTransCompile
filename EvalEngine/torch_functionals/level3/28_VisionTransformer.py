import torch
import torch.nn as nn
import torch.nn.functional as F

def transformer_encoder_layer_forward(x, self_attn, linear1, linear2, norm1, norm2, dropout, activation):
    x = x + self_attn(x, x, x)[0]
    x = norm1(x)
    x2 = linear2(activation(linear1(x)))
    x = x + dropout(x2)
    x = norm2(x)
    return x

def transformer_encoder_forward(x, layers, num_layers):
    for layer in layers:
        x = layer(x)
    return x

def module_fn(img, patch_size, pos_embedding, patch_to_embedding_weight, patch_to_embedding_bias, cls_token, dropout_p, transformer_layers, mlp_head_0_weight, mlp_head_0_bias, mlp_head_3_weight, mlp_head_3_bias):
    p = patch_size
    
    x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
    x = F.linear(x, patch_to_embedding_weight, patch_to_embedding_bias)
    
    cls_tokens = cls_token.expand(img.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x += pos_embedding
    x = F.dropout(x, p=dropout_p, training=False)
    
    for layer in transformer_layers:
        x = transformer_encoder_layer_forward(
            x,
            layer.self_attn,
            layer.linear1,
            layer.linear2,
            layer.norm1,
            layer.norm2,
            layer.dropout2,
            F.gelu
        )
    
    x = x[:, 0]
    x = F.linear(x, mlp_head_0_weight, mlp_head_0_bias)
    x = F.gelu(x)
    x = F.dropout(x, p=dropout_p, training=False)
    x = F.linear(x, mlp_head_3_weight, mlp_head_3_bias)
    return x

class Model(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(Model, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.dim = dim
        self.heads = heads
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding_weight = nn.Parameter(torch.randn(dim, patch_dim))
        self.patch_to_embedding_bias = nn.Parameter(torch.randn(dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_p = emb_dropout
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.mlp_head_0_weight = nn.Parameter(torch.randn(mlp_dim, dim))
        self.mlp_head_0_bias = nn.Parameter(torch.randn(mlp_dim))
        self.mlp_head_3_weight = nn.Parameter(torch.randn(num_classes, mlp_dim))
        self.mlp_head_3_bias = nn.Parameter(torch.randn(num_classes))
    
    def forward(self, img, fn=module_fn):
        if fn == module_fn:
            # 原生 PyTorch 前向传播
            return fn(
                img,
                self.patch_size,
                self.pos_embedding,
                self.patch_to_embedding_weight.detach(),
                self.patch_to_embedding_bias,
                self.cls_token,
                self.dropout_p,
                self.transformer_layers,
                self.mlp_head_0_weight.detach(),
                self.mlp_head_0_bias,
                self.mlp_head_3_weight.detach(),
                self.mlp_head_3_bias
            )
        else:
            # CUDA 扩展前向传播
            args = [
                img,
                self.patch_size,
                self.pos_embedding.detach(),
                self.patch_to_embedding_weight.detach(),
                self.patch_to_embedding_bias.detach(),
                self.cls_token.detach(),
                self.dropout_p,
            ]
            # 解包 Transformer 层参数
            for layer in self.transformer_layers:
                args.extend([
                    layer.self_attn.in_proj_weight.detach(),
                    layer.self_attn.in_proj_bias.detach(),
                    layer.self_attn.out_proj.weight.detach(),
                    layer.self_attn.out_proj.bias.detach(),
                    layer.linear1.weight.detach(),
                    layer.linear1.bias.detach(),
                    layer.linear2.weight.detach(),
                    layer.linear2.bias.detach(),
                    layer.norm1.weight.detach(),
                    layer.norm1.bias.detach(),
                    layer.norm2.weight.detach(),
                    layer.norm2.bias.detach(),
                ])
            # 添加 dim 和 heads
            args.extend([self.dim, self.heads])
            # 添加 mlp_head 参数
            args.extend([
                self.mlp_head_0_weight.detach(),
                self.mlp_head_0_bias.detach(),
                self.mlp_head_3_weight.detach(),
                self.mlp_head_3_bias.detach(),
            ])
            return fn(*args)

# Test code
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    return [torch.randn(2, channels, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]
```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function for layer normalization
__global__ void layer_norm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int num_tokens,
    int embed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * embed_dim) {
        int token_idx = idx / embed_dim;
        int dim_idx = idx % embed_dim;
        
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < embed_dim; ++i) {
            mean += input[token_idx * embed_dim + i];
        }
        mean /= embed_dim;
        
        // Compute variance
        float var = 0.0f;
        for (int i = 0; i < embed_dim; ++i) {
            float diff = input[token_idx * embed_dim + i] - mean;
            var += diff * diff;
        }
        var /= embed_dim;
        
        // Normalize
        float inv_std = rsqrtf(var + 1e-5f);
        output[idx] = (input[idx] - mean) * inv_std * weight[dim_idx] + bias[dim_idx];
    }
}

// Utility function for matrix multiplication
__global__ void matmul_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_dim,
    int output_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum += input[row * input_dim + i] * weight[i * output_dim + col];
        }
        output[row * output_dim + col] = sum + (bias ? bias[col] : 0.0f);
    }
}

// Utility function for GELU activation
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Main forward function
std::vector<torch::Tensor> forward(
    torch::Tensor x,
    torch::Tensor conv1_weight,
    torch::Tensor conv1_bias,
    torch::Tensor linear_proj_weight,
    torch::Tensor linear_proj_bias,
    torch::Tensor cls_token,
    torch::Tensor fc_out_weight,
    torch::Tensor fc_out_bias,
    torch::Tensor transformer_layers_self_attn_in_proj_weight,
    torch::Tensor transformer_layers_self_attn_in_proj_bias,
    torch::Tensor transformer_layers_self_attn_out_proj_weight,
    torch::Tensor transformer_layers_self_attn_out_proj_bias,
    torch::Tensor transformer_layers_linear1_weight,
    torch::Tensor transformer_layers_linear1_bias,
    torch::Tensor transformer_layers_linear2_weight,
    torch::Tensor transformer_layers_linear2_bias,
    torch::Tensor transformer_layers_norm1_weight,
    torch::Tensor transformer_layers_norm1_bias,
    torch::Tensor transformer_layers_norm2_weight,
    torch::Tensor transformer_layers_norm2_bias,
    int num_heads,
    int num_layers,
    int patch_size,
    int embed_dim,
    float mlp_ratio
) {
    // Check inputs
    CHECK_INPUT(x);
    CHECK_INPUT(conv1_weight);
    CHECK_INPUT(conv1_bias);
    CHECK_INPUT(linear_proj_weight);
    CHECK_INPUT(linear_proj_bias);
    CHECK_INPUT(cls_token);
    CHECK_INPUT(fc_out_weight);
    CHECK_INPUT(fc_out_bias);
    CHECK_INPUT(transformer_layers_self_attn_in_proj_weight);
    CHECK_INPUT(transformer_layers_self_attn_in_proj_bias);
    CHECK_INPUT(transformer_layers_self_attn_out_proj_weight);
    CHECK_INPUT(transformer_layers_self_attn_out_proj_bias);
    CHECK_INPUT(transformer_layers_linear1_weight);
    CHECK_INPUT(transformer_layers_linear1_bias);
    CHECK_INPUT(transformer_layers_linear2_weight);
    CHECK_INPUT(transformer_layers_linear2_bias);
    CHECK_INPUT(transformer_layers_norm1_weight);
    CHECK_INPUT(transformer_layers_norm1_bias);
    CHECK_INPUT(transformer_layers_norm2_weight);
    CHECK_INPUT(transformer_layers_norm2_bias);

    // Get dimensions
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    // Convolutional patch embedding
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto x_conv = torch::conv2d(x, conv1_weight, conv1_bias, patch_size);
    x_conv = x_conv.flatten(1);
    
    // Linear projection
    auto x_proj = torch::matmul(x_conv, linear_proj_weight.t()) + linear_proj_bias;
    
    // Add cls token
    auto cls_tokens = cls_token.expand({B, 1, embed_dim});
    x_proj = x_proj.view({B, -1, embed_dim});
    x_proj = torch::cat({cls_tokens, x_proj}, 1);
    
    // Transformer layers
    for (int i = 0; i < num_layers; ++i) {
        // Self-attention
        auto in_proj_weight = transformer_layers_self_attn_in_proj_weight[i];
        auto in_proj_bias = transformer_layers_self_attn_in_proj_bias[i];
        auto out_proj_weight = transformer_layers_self_attn_out_proj_weight[i];
        auto out_proj_bias = transformer_layers_self_attn_out_proj_bias[i];
        
        // QKV projection
        auto qkv = torch::matmul(x_proj, in_proj_weight.t()) + in_proj_bias;
        auto q = qkv.slice(2, 0, embed_dim);
        auto k = qkv.slice(2, embed_dim, 2*embed_dim);
        auto v = qkv.slice(2, 2*embed_dim, 3*embed_dim);
        
        // Scaled dot-product attention
        auto attn = torch::matmul(q, k.transpose(1, 2)) / sqrt(embed_dim / num_heads);
        attn = torch::softmax(attn, -1);
        auto attn_output = torch::matmul(attn, v);
        
        // Output projection
        attn_output = torch::matmul(attn_output, out_proj_weight.t()) + out_proj_bias;
        
        // Residual connection and norm
        x_proj = x_proj + attn_output;
        x_proj = torch::layer_norm(x_proj, {embed_dim}, 
                                 transformer_layers_norm1_weight[i], 
                                 transformer_layers_norm1_bias[i]);
        
        // Feedforward network
        auto linear1_weight = transformer_layers_linear1_weight[i];
        auto linear1_bias = transformer_layers_linear1_bias[i];
        auto linear2_weight = transformer_layers_linear2_weight[i];
        auto linear2_bias = transformer_layers_linear2_bias[i];
        
        auto ff = torch::matmul(x_proj, linear1_weight.t()) + linear1_bias;
        ff = torch::gelu(ff);
        ff = torch::matmul(ff, linear2_weight.t()) + linear2_bias;
        
        // Residual connection and norm
        x_proj = x_proj + ff;
        x_proj = torch::layer_norm(x_proj, {embed_dim}, 
                                 transformer_layers_norm2_weight[i], 
                                 transformer_layers_norm2_bias[i]);
    }
    
    // Classify based on cls token
    auto cls_output = x_proj.slice(1, 0, 1).squeeze(1);
    auto output = torch::matmul(cls_output, fc_out_weight.t()) + fc_out_bias;
    
    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vision Transformer forward");
}
```
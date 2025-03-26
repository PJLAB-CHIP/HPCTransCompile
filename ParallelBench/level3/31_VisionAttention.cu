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

// Kernel for in_projection_packed
__global__ void in_projection_packed_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t input_size,
    int64_t output_size,
    int64_t batch_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        int64_t batch = idx / output_size;
        int64_t out_feature = idx % output_size;
        
        float sum = bias ? bias[out_feature] : 0.0f;
        for (int64_t in_feature = 0; in_feature < input_size; ++in_feature) {
            sum += input[batch * input_size + in_feature] * 
                   weight[out_feature * input_size + in_feature];
        }
        output[idx] = sum;
    }
}

// Kernel for attention computation
__global__ void attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* attn_output,
    float* attn_weights,
    int64_t seq_len,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dim,
    float scaling
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_positions = batch_size * num_heads * seq_len * seq_len;
    
    if (idx < total_positions) {
        int64_t b = idx / (num_heads * seq_len * seq_len);
        int64_t h = (idx / (seq_len * seq_len)) % num_heads;
        int64_t i = (idx / seq_len) % seq_len;
        int64_t j = idx % seq_len;
        
        float sum = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
            sum += q[(b * num_heads + h) * seq_len * head_dim + i * head_dim + d] * 
                   k[(b * num_heads + h) * seq_len * head_dim + j * head_dim + d];
        }
        attn_weights[idx] = sum * scaling;
    }
}

// Kernel for softmax
__global__ void softmax_kernel(
    float* attn_weights,
    int64_t seq_len,
    int64_t batch_size,
    int64_t num_heads
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_heads = batch_size * num_heads * seq_len;
    
    if (idx < total_heads) {
        int64_t b = idx / (num_heads * seq_len);
        int64_t h = (idx / seq_len) % num_heads;
        int64_t i = idx % seq_len;
        
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int64_t j = 0; j < seq_len; ++j) {
            float val = attn_weights[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j];
            if (val > max_val) max_val = val;
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int64_t j = 0; j < seq_len; ++j) {
            float val = expf(attn_weights[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j] - max_val);
            sum += val;
            attn_weights[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j] = val;
        }
        
        // Normalize
        for (int64_t j = 0; j < seq_len; ++j) {
            attn_weights[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j] /= sum;
        }
    }
}

// Kernel for attention output
__global__ void attention_output_kernel(
    const float* attn_weights,
    const float* v,
    float* output,
    int64_t seq_len,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dim
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_positions = batch_size * num_heads * seq_len * head_dim;
    
    if (idx < total_positions) {
        int64_t b = idx / (num_heads * seq_len * head_dim);
        int64_t h = (idx / (seq_len * head_dim)) % num_heads;
        int64_t i = (idx / head_dim) % seq_len;
        int64_t d = idx % head_dim;
        
        float sum = 0.0f;
        for (int64_t j = 0; j < seq_len; ++j) {
            sum += attn_weights[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j] * 
                   v[(b * num_heads + h) * seq_len * head_dim + j * head_dim + d];
        }
        output[idx] = sum;
    }
}

// Kernel for layer norm
__global__ void layer_norm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t num_features,
    int64_t batch_size,
    int64_t seq_len,
    float eps
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * seq_len;
    
    if (idx < total_elements) {
        int64_t b = idx / seq_len;
        int64_t i = idx % seq_len;
        
        // Compute mean
        float mean = 0.0f;
        for (int64_t f = 0; f < num_features; ++f) {
            mean += input[b * seq_len * num_features + i * num_features + f];
        }
        mean /= num_features;
        
        // Compute variance
        float var = 0.0f;
        for (int64_t f = 0; f < num_features; ++f) {
            float diff = input[b * seq_len * num_features + i * num_features + f] - mean;
            var += diff * diff;
        }
        var /= num_features;
        
        // Normalize
        for (int64_t f = 0; f < num_features; ++f) {
            output[b * seq_len * num_features + i * num_features + f] = 
                (input[b * seq_len * num_features + i * num_features + f] - mean) / sqrtf(var + eps) * weight[f] + bias[f];
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t embed_dim,
    int64_t num_heads,
    torch::Tensor in_proj_weight,
    torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(in_proj_weight);
    CHECK_INPUT(in_proj_bias);
    CHECK_INPUT(out_proj_weight);
    CHECK_INPUT(out_proj_bias);
    CHECK_INPUT(norm_weight);
    CHECK_INPUT(norm_bias);
    
    at::cuda::CUDAGuard device_guard(x.device());
    
    auto B = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto seq_len = H * W;
    auto head_dim = embed_dim / num_heads;
    auto scaling = powf(head_dim, -0.5f);
    
    // Reshape input
    x = x.view({B, C, seq_len}).permute({2, 0, 1}).contiguous();
    
    // Allocate intermediate tensors
    auto qkv = torch::empty({3 * seq_len * B * embed_dim}, x.options());
    auto q = qkv.slice(0, 0, seq_len * B * embed_dim);
    auto k = qkv.slice(0, seq_len * B * embed_dim, 2 * seq_len * B * embed_dim);
    auto v = qkv.slice(0, 2 * seq_len * B * embed_dim, 3 * seq_len * B * embed_dim);
    
    // Compute Q, K, V projections
    {
        dim3 blocks((seq_len * B * embed_dim + 255) / 256, 1, 1);
        dim3 threads(256, 1, 1);
        in_projection_packed_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            in_proj_weight.data_ptr<float>(),
            in_proj_bias.data_ptr<float>(),
            qkv.data_ptr<float>(),
            embed_dim,
            3 * embed_dim,
            seq_len * B
        );
    }
    
    // Reshape Q, K, V for attention
    q = q.view({seq_len, B, num_heads, head_dim}).permute({1, 2, 0, 3}).contiguous();
    k = k.view({seq_len, B, num_heads, head_dim}).permute({1, 2, 0, 3}).contiguous();
    v = v.view({seq_len, B, num_heads, head_dim}).permute({1, 2, 0, 3}).contiguous();
    
    // Scale Q
    q = q * scaling;
    
    // Compute attention weights
    auto attn_weights = torch::empty({B * num_heads * seq_len * seq_len}, x.options());
    {
        dim3 blocks((B * num_heads * seq_len * seq_len + 255) / 256, 1, 1);
        dim3 threads(256, 1, 1);
        attention_kernel<<<blocks, threads>>>(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            nullptr,
            attn_weights.data_ptr<float>(),
            seq_len,
            B,
            num_heads,
            head_dim,
            scaling
        );
    }
    
    // Apply softmax
    {
        dim3 blocks((B * num_heads * seq_len + 255) / 256, 1, 1);
        dim3 threads(256, 1, 1);
        softmax_kernel<<<blocks, threads>>>(
            attn_weights.data_ptr<float>(),
            seq_len,
            B,
            num_heads
        );
    }
    
    // Compute attention output
    auto attn_output = torch::empty({B * num_heads * seq_len * head_dim}, x.options());
    {
        dim3 blocks((B * num_heads * seq_len * head_dim + 255) / 256, 1, 1);
        dim3 threads(256, 1, 1);
        attention_output_kernel<<<blocks, threads>>>(
            attn_weights.data_ptr<float>(),
            v.data_ptr<float>(),
            attn_output.data_ptr<float>(),
            seq_len,
            B,
            num_heads,
            head_dim
        );
    }
    
    // Reshape attention output
    attn_output = attn_output.view({B, num_heads, seq_len, head_dim})
                 .permute({2, 0, 1, 3}).contiguous()
                 .view({seq_len, B, embed_dim});
    
    // Apply output projection
    auto output = torch::empty({seq_len, B, embed_dim}, x.options());
    {
        dim3 blocks((seq_len * B * embed_dim + 255) / 256, 1, 1);
        dim3 threads(256, 1, 1);
        in_projection_packed_kernel<<<blocks, threads>>>(
            attn_output.data_ptr<float>(),
            out_proj_weight.data_ptr<float>(),
            out_proj_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            embed_dim,
            embed_dim,
            seq_len * B
        );
    }
    
    // Add residual and apply layer norm
    output = output + x;
    auto norm_output = torch::empty_like(output);
    {
        dim3 blocks((B * seq_len + 255) / 256, 1, 1);
        dim3 threads(256, 1, 1);
        layer_norm_kernel<<<blocks, threads>>>(
            output.data_ptr<float>(),
            norm_weight.data_ptr<float>(),
            norm_bias.data_ptr<float>(),
            norm_output.data_ptr<float>(),
            embed_dim,
            B,
            seq_len,
            1e-5f
        );
    }
    
    // Reshape back to original dimensions
    norm_output = norm_output.permute({1, 2, 0}).contiguous().view({B, C, H, W});
    
    return norm_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom CUDA forward");
}
```
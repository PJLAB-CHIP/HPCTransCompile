#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); 
namespace py = pybind11;
__global__ void layer_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int num_features,
    int num_elements,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int batch_idx = idx / num_features;
        int feat_idx = idx % num_features;
        
        // 双精度计算均值方差
        double mean = 0.0;
        for (int i = 0; i < num_features; ++i) {
            mean += (double)input[batch_idx * num_features + i];
        }
        mean /= num_features;
        
        double var = 0.0;
        for (int i = 0; i < num_features; ++i) {
            double diff = (double)input[batch_idx * num_features + i] - mean;
            var += diff * diff;
        }
        var = var / num_features + (double)eps;
        
        // 归一化
        double norm_val = ((double)input[batch_idx * num_features + feat_idx] - mean) / sqrt(var);
        output[batch_idx * num_features + feat_idx] = (float)(norm_val * (double)gamma[feat_idx] + (double)beta[feat_idx]);
    }
}

// 自定义LayerNorm函数
torch::Tensor custom_layer_norm(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps = 1e-5
) {
    // 检查输入
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    // 获取维度信息
    const int num_features = gamma.size(0);    // embed_dim
    const int num_elements = input.numel();     // B*seq_len*embed_dim
    
    // 创建输出张量
    auto output = torch::empty_like(input);
    
    // 配置CUDA执行参数
    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    // 启动内核
    layer_norm_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        num_features,
        num_elements,
        eps
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    
    return output;
}

// my_matmul_kernel.cu

__global__ void matrix_multiply_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int a_stride0, int a_stride1,
    int b_stride0, int b_stride1,
    int c_stride0, int c_stride1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * a_stride0 + k * a_stride1] * 
                   B[k * b_stride0 + col * b_stride1];
        }
        C[row * c_stride0 + col * c_stride1] = sum;
    }
}

torch::Tensor my_matmul(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    // 获取维度信息
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    // 创建输出张量
    auto C = torch::zeros({M, N}, A.options());

    // 配置CUDA执行参数
    dim3 threads(16, 16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    // 获取步长信息
    int a_stride0 = A.stride(0), a_stride1 = A.stride(1);
    int b_stride0 = B.stride(0), b_stride1 = B.stride(1);
    int c_stride0 = C.stride(0), c_stride1 = C.stride(1);

    // 启动内核
    matrix_multiply_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        a_stride0, a_stride1,
        b_stride0, b_stride1,
        c_stride0, c_stride1
    );

    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

// Utility function for GELU activation
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

torch::Tensor scaled_dot_product_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, float dropout_prob) {
    // Calculate attention scores (scaled dot product)
    auto attn_scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(q.size(-1));
    // Apply softmax to get attention weights
    auto attn_weights = torch::softmax(attn_scores, -1);
    
    // Optionally apply dropout (if required)
    if (dropout_prob > 0.0) {
        attn_weights = torch::dropout(attn_weights, dropout_prob, /*train=*/true);
    }

    // Calculate attention output
    auto attn_output = torch::matmul(attn_weights, v);

    return attn_output;
}

// Main forward function
torch::Tensor forward(
    torch::Tensor x,
    py::object params_obj,
    int num_heads,
    int num_layers,
    int patch_size,
    int embed_dim,
    float mlp_ratio
) {
    // Check inputs
    CHECK_INPUT(x);
    // Convert params to a Python dictionary if necessary
    py::dict params = params_obj.cast<py::dict>();
    // 对齐PyTorch的参数初始化
    torch::Tensor conv1_weight = params["conv1_weight"].cast<torch::Tensor>();
    torch::Tensor conv1_bias = params["conv1_bias"].cast<torch::Tensor>();
    torch::Tensor linear_proj_weight = params["linear_proj_weight"].cast<torch::Tensor>();
    torch::Tensor linear_proj_bias = params["linear_proj_bias"].cast<torch::Tensor>();
    torch::Tensor cls_token = params["cls_token"].cast<torch::Tensor>();
    torch::Tensor fc_out_weight = params["fc_out_weight"].cast<torch::Tensor>();
    torch::Tensor fc_out_bias = params["fc_out_bias"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_self_attn_in_proj_weight = params["transformer_layers_self_attn_in_proj_weight"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_self_attn_in_proj_bias = params["transformer_layers_self_attn_in_proj_bias"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_self_attn_out_proj_weight = params["transformer_layers_self_attn_out_proj_weight"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_self_attn_out_proj_bias = params["transformer_layers_self_attn_out_proj_bias"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_linear1_weight = params["transformer_layers_linear1_weight"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_linear1_bias = params["transformer_layers_linear1_bias"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_linear2_weight = params["transformer_layers_linear2_weight"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_linear2_bias = params["transformer_layers_linear2_bias"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_norm1_weight = params["transformer_layers_norm1_weight"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_norm1_bias = params["transformer_layers_norm1_bias"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_norm2_weight = params["transformer_layers_norm2_weight"].cast<torch::Tensor>();
    torch::Tensor transformer_layers_norm2_bias = params["transformer_layers_norm2_bias"].cast<torch::Tensor>();
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
    auto x_proj = my_matmul(x_conv, linear_proj_weight.t()) + linear_proj_bias;
    
    // Add cls token
    auto cls_tokens = cls_token.expand({B, 1, embed_dim});
    // x_proj = x_proj.view({B, -1, embed_dim});
    x_proj = x_proj.unsqueeze(1);
    x_proj = torch::cat({cls_tokens, x_proj}, 1);
    // return x_proj;
    // Transformer layers
    for (int i = 0; i < 1; ++i) {
        // Self-attention
        // 提取当前层的参数
        auto in_proj_weight = transformer_layers_self_attn_in_proj_weight[i];
        auto in_proj_bias = transformer_layers_self_attn_in_proj_bias[i];
        auto out_proj_weight = transformer_layers_self_attn_out_proj_weight[i];
        auto out_proj_bias = transformer_layers_self_attn_out_proj_bias[i];
        
        // QKV projection

        auto qkv = torch::matmul(x_proj, in_proj_weight.t()) + in_proj_bias; // [B, seq_len, 3*embed_dim]
        auto qkv_chunks = qkv.chunk(3, /*dim=*/-1); // 分割为q, k, v各[B, seq_len, embed_dim]
        auto q = qkv_chunks[0], k = qkv_chunks[1], v = qkv_chunks[2];
        auto attn_output = scaled_dot_product_attention(q, k, v, 0.0);

        // 输出投影
        attn_output = torch::matmul(attn_output, out_proj_weight.t()) + out_proj_bias; // [B, seq_len, embed_dim]

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
    auto output = my_matmul(cls_output, fc_out_weight.t()) + fc_out_bias;
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vision Transformer forward");
}
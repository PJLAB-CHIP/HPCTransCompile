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
#define CUDA_CHECK(call) { cudaError_t _e = (call); if(_e != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(_e)); }
namespace py = pybind11;

// LayerNorm核函数（双精度计算）
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


torch::Tensor vision_attention_forward(torch::Tensor x, py::object params_obj, int64_t embed_dim, int64_t num_heads) {
    CHECK_INPUT(x);
    // Convert params to a Python dictionary if necessary
    py::dict params = params_obj.cast<py::dict>();
    // 固定参数
    const int64_t head_dim = embed_dim / num_heads;
    const float scaling_factor = 1.0f / sqrtf(static_cast<float>(head_dim));
    at::cuda::CUDAGuard guard(x.device());
    // auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    
    // 对齐PyTorch的参数初始化
    torch::Tensor in_proj_weight = params["in_proj_weight"].cast<torch::Tensor>();
    torch::Tensor in_proj_bias = params["in_proj_bias"].cast<torch::Tensor>();
    torch::Tensor out_proj_weight = params["out_proj_weight"].cast<torch::Tensor>();
    torch::Tensor out_proj_bias = params["out_proj_bias"].cast<torch::Tensor>();
    torch::Tensor norm_weight = params["norm_weight"].cast<torch::Tensor>();
    torch::Tensor norm_bias = params["norm_bias"].cast<torch::Tensor>();
    
    // 输入形状处理
    auto sizes = x.sizes();
    int64_t B = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    int64_t seq_len = H * W; // 16384
    auto x_reshaped = x.view({B, C, H * W}).permute({2, 0, 1}).contiguous(); // seq, batchsize, headNums

    // 使用 linear 函数分别计算 Q、K 和 V 的投影
    auto q = torch::nn::functional::linear(x_reshaped, in_proj_weight.slice(0, 0, embed_dim), in_proj_bias.slice(0, 0, embed_dim));
    auto k = torch::nn::functional::linear(x_reshaped, in_proj_weight.slice(0, embed_dim, 2 * embed_dim), in_proj_bias.slice(0, embed_dim, 2 * embed_dim));
    auto v = torch::nn::functional::linear(x_reshaped, in_proj_weight.slice(0, 2 * embed_dim, 3 * embed_dim), in_proj_bias.slice(0, 2 * embed_dim, 3 * embed_dim));
    q = q.mul_(scaling_factor);  
    q = q.contiguous().view({-1, q.size(1), num_heads, head_dim}).transpose(1, 2);
    k = k.contiguous().view({-1, k.size(1), num_heads, head_dim}).transpose(1, 2);
    v = v.contiguous().view({-1, v.size(1), num_heads, head_dim}).transpose(1, 2);

    // 2. 计算注意力权重（q * k^T）
    auto attn_output_weights = torch::matmul(q, k.transpose(-2, -1));

    // 3. softmax（注意力权重）
    attn_output_weights = torch::softmax(attn_output_weights, /*dim=*/-1);

    // 4. 计算最终输出（注意力权重 * v）
    auto attn_output = torch::matmul(attn_output_weights, v);

    // 5. 转置和调整形状（将其恢复为原始形状）
    attn_output = attn_output.transpose(1, 2).contiguous().view({attn_output.size(0), -1, embed_dim});

    // 6. 线性变换
    attn_output = torch::linear(attn_output, out_proj_weight, out_proj_bias);
    // 检查权重/偏置形状
    TORCH_CHECK(norm_weight.sizes() == torch::IntArrayRef({embed_dim}), 
    "Weight shape mismatch");
    TORCH_CHECK(norm_bias.sizes() == torch::IntArrayRef({embed_dim}), 
    "Bias shape mismatch");

    // 检查输入维度
    TORCH_CHECK((attn_output + x_reshaped).size(-1) == embed_dim, 
    "Last dimension must be embed_dim");
    // 7. 残差连接 + LayerNorm
    auto residual = attn_output + x_reshaped;
    auto norm_output = torch::empty_like(residual);
    {
         int num_elements = seq_len * B * embed_dim;
         dim3 blocks((num_elements + 255) / 256);
        dim3 threads(256);
        layer_norm_kernel<<<blocks, threads>>>(
            residual.data_ptr<float>(),
            norm_weight.data_ptr<float>(),
            norm_bias.data_ptr<float>(),
            norm_output.data_ptr<float>(),
            embed_dim,
            num_elements,
            1e-5f
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // 恢复原始形状
    return norm_output.permute({1, 2, 0}).view({B, C, H, W}).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vision_attention_forward, "Vision Attention Forward");
}
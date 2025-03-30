#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

// Optimized block sizes based on H100 architecture
constexpr int BLOCK_SIZE_GELU = 128;  // Optimized for GELU computation
constexpr int VECTOR_SIZE = 4;        // float4 vectorization
constexpr int SHARED_MEM_ELEMENTS = BLOCK_SIZE_GELU * VECTOR_SIZE;

__forceinline__ float gelu_activation(float x) {
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    return x * 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

torch::Tensor optimized_block_gelu_kernel(
    const torch::Tensor& input,
    const int size) {
    
    torch::Tensor output = torch::empty_like(input);
    int total_elements = size;
    int num_blocks = (total_elements / (VECTOR_SIZE * BLOCK_SIZE_GELU)) + 1;
    
    #pragma omp parallel for
    for (int idx = 0; idx < total_elements; idx += VECTOR_SIZE) {
        // Load data into shared memory using float4
        float4 in_val;
        in_val.x = input[idx];
        in_val.y = input[idx + 1];
        in_val.z = input[idx + 2];
        in_val.w = input[idx + 3];
        
        // Process data in shared memory
        float out_val = gelu_activation(in_val.x);
        out_val = gelu_activation(in_val.y);
        out_val = gelu_activation(in_val.z);
        out_val = gelu_activation(in_val.w);
        
        // Store results
        output[idx] = out_val;
        output[idx + 1] = out_val;
        output[idx + 2] = out_val;
        output[idx + 3] = out_val;
    }
    
    return output;
}

torch::Tensor forward(
    const torch::Tensor& x,
    const py::object& params,
    bool is_training) {
    
    py::dict params_dict = params.cast<py::dict>();
    
    auto n_embd = params_dict["n_embd"].cast<int64_t>();
    auto n_head = params_dict["n_head"].cast<int64_t>();
    auto attn_pdrop = params_dict["attn_pdrop"].cast<float>();
    auto resid_pdrop = params_dict["resid_pdrop"].cast<float>();
    
    auto ln1_weight = params_dict["ln1_weight"].cast<torch::Tensor>();
    auto ln1_bias = params_dict["ln1_bias"].cast<torch::Tensor>();
    auto c_attn_weight = params_dict["c_attn_weight"].cast<torch::Tensor>();
    auto c_attn_bias = params_dict["c_attn_bias"].cast<torch::Tensor>();
    auto c_proj_weight = params_dict["c_proj_weight"].cast<torch::Tensor>();
    auto c_proj_bias = params_dict["c_proj_bias"].cast<torch::Tensor>();
    auto bias = params_dict["bias"].cast<torch::Tensor>();
    auto ln2_weight = params_dict["ln2_weight"].cast<torch::Tensor>();
    auto ln2_bias = params_dict["ln2_bias"].cast<torch::Tensor>();
    auto mlp_fc_weight = params_dict["mlp_fc_weight"].cast<torch::Tensor>();
    auto mlp_fc_bias = params_dict["mlp_fc_bias"].cast<torch::Tensor>();
    auto mlp_proj_weight = params_dict["mlp_proj_weight"].cast<torch::Tensor>();
    auto mlp_proj_bias = params_dict["mlp_proj_bias"].cast<torch::Tensor>();
    
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);
    const int64_t head_size = C / n_head;
    
    // Layer norm 1
    auto ln1_out = torch::layer_norm(x, {n_embd}, ln1_weight, ln1_bias);
    
    // Self-attention
    auto qkv = torch::linear(ln1_out, c_attn_weight, c_attn_bias);
    auto qkv_split = qkv.chunk(3, /*dim=*/2);
    auto q = qkv_split[0].view({B, T, n_head, head_size}).transpose(1, 2);
    auto k = qkv_split[1].view({B, T, n_head, head_size}).transpose(1, 2);
    auto v = qkv_split[2].view({B, T, n_head, head_size}).transpose(1, 2);
    
    auto att = torch::matmul(q, k.transpose(-2, -1)) * (1.0f / sqrt(static_cast<float>(head_size)));
    att = att.masked_fill(bias.slice(2, 0, T).slice(3, 0, T) == 0, -INFINITY);
    att = torch::softmax(att, -1);
    
    if (is_training) {
        att = torch::dropout(att, attn_pdrop, true);
    }
    
    auto y = torch::matmul(att, v);
    y = y.transpose(1, 2).contiguous().view({B, T, C});
    auto attn_out = torch::linear(y, c_proj_weight, c_proj_bias);
    if (is_training) {
        attn_out = torch::dropout(attn_out, resid_pdrop, true);
    }
    
    auto x_plus_attn_out = x + attn_out;
    
    // MLP block
    auto ln2_out = torch::layer_norm(x_plus_attn_out, {n_embd}, ln2_weight, ln2_bias);
    auto fc_out = torch::linear(ln2_out, mlp_fc_weight, mlp_fc_bias);
    
    // Apply optimized block size GELU
    auto gelu_out = fc_out.clone();
    int total_elements = fc_out.numel();
    auto gelu_out_reshaped = optimized_block_gelu_kernel(gelu_out, total_elements);
    
    auto proj_out = torch::linear(gelu_out_reshaped, mlp_proj_weight, mlp_proj_bias);
    if (is_training) {
        proj_out = torch::dropout(proj_out, resid_pdrop, true);
    }
    
    return x_plus_attn_out + proj_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Block size optimized transformer block forward (CPU)");
}
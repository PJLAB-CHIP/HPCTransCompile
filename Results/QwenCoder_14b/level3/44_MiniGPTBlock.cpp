#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

constexpr int BLOCK_SIZE_GELU = 128;  // Optimized for GELU computation
constexpr int VECTOR_SIZE = 4;        // float4 vectorization

__forceinline__ float gelu_activation(float x) {
    const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
    return x * 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

void optimized_block_gelu_cpu(
    float* output,
    const float* input,
    const int size) {
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int idx = 0; idx < size / VECTOR_SIZE; ++idx) {
        // Load data into local variables using float4
        float4 in_val;
        in_val.x = input[idx * VECTOR_SIZE + 0];
        in_val.y = input[idx * VECTOR_SIZE + 1];
        in_val.z = input[idx * VECTOR_SIZE + 2];
        in_val.w = input[idx * VECTOR_SIZE + 3];
        
        // Process data
        float4 out_val;
        out_val.x = gelu_activation(in_val.x);
        out_val.y = gelu_activation(in_val.y);
        out_val.z = gelu_activation(in_val.z);
        out_val.w = gelu_activation(in_val.w);
        
        // Store results
        output[idx * VECTOR_SIZE + 0] = out_val.x;
        output[idx * VECTOR_SIZE + 1] = out_val.y;
        output[idx * VECTOR_SIZE + 2] = out_val.z;
        output[idx * VECTOR_SIZE + 3] = out_val.w;
    }
    
    // Handle remaining elements
    #pragma omp parallel for
    for (int rem_idx = (size / (VECTOR_SIZE * BLOCK_SIZE_GELU)) * (VECTOR_SIZE * BLOCK_SIZE_GELU); rem_idx < size; ++rem_idx) {
        output[rem_idx] = gelu_activation(input[rem_idx]);
    }
}

torch::Tensor forward(
    torch::Tensor x,
    py::object params,
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
    
    x = x + attn_out;
    
    // MLP block
    auto ln2_out = torch::layer_norm(x, {n_embd}, ln2_weight, ln2_bias);
    auto fc_out = torch::linear(ln2_out, mlp_fc_weight, mlp_fc_bias);
    
    // Apply optimized block size GELU
    auto gelu_out = fc_out.clone();
    int total_elements = fc_out.numel();
    
    optimized_block_gelu_cpu(
        gelu_out.data_ptr<float>(),
        fc_out.data_ptr<float>(),
        total_elements
    );
    
    auto proj_out = torch::linear(gelu_out, mlp_proj_weight, mlp_proj_bias);
    if (is_training) {
        proj_out = torch::dropout(proj_out, resid_pdrop, true);
    }
    
    return x + proj_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Block size optimized transformer block forward (CPU)");
}
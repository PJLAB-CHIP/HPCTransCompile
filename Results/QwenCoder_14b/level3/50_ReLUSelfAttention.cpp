#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>
#include <limits>

// Function to apply shared bias kernel on CPU
void shared_bias_cpu(
    float* att,
    const float* bias,
    int64_t B,
    int64_t n_head,
    int64_t T,
    float scale,
    float fill_value
) {
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t i = 0; i < T; ++i) {
                for (int64_t j = 0; j < T; ++j) {
                    int64_t idx = b * n_head * T * T + h * T * T + i * T + j;
                    float val = att[idx] * scale;
                    bool is_masked = (bias[i * T + j] == 0.0f);
                    val = is_masked ? fill_value : val;
                    att[idx] = std::max(val, 0.0f);
                }
            }
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor c_attn_weight,
    at::Tensor c_attn_bias,
    at::Tensor bias,
    int64_t n_head,
    int64_t n_embd
) {
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);
    const int64_t hs = C / n_head;
    const float scale = 1.0f / std::sqrt(hs);

    // Compute qkv projections
    at::Tensor qkv = at::addmm(c_attn_bias, x.view({B*T, C}), c_attn_weight.t()).view({B, T, 3*C});
    
    // Split and reshape q,k,v
    auto chunks = qkv.split({C, C, C}, 2);
    at::Tensor q = chunks[0].view({B, T, n_head, hs}).permute({0, 2, 1, 3});
    at::Tensor k = chunks[1].view({B, T, n_head, hs}).permute({0, 2, 1, 3});
    
    // Compute attention matrix
    at::Tensor att = at::matmul(q, k.transpose(-2, -1)).contiguous();

    // Prepare bias slice
    at::Tensor bias_slice = bias.slice(2, 0, T).slice(3, 0, T).contiguous();
    const float* bias_data = bias_slice.data_ptr<float>();

    // Apply shared bias kernel on CPU
    shared_bias_cpu(
        att.data_ptr<float>(),
        bias_data,
        B,
        n_head,
        T,
        scale,
        -std::numeric_limits<float>::infinity()
    );

    // Final matmul and reshape
    return at::matmul(att, chunks[2].view({B, T, n_head, hs}).permute({0, 2, 1, 3}))
           .permute({0, 2, 1, 3}).reshape({B, T, C});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Bias 50_ReLUSelfAttention");
}
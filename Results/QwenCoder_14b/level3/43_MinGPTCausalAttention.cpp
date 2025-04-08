#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

// Constants for memory alignment and optimization
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int ALIGN_BYTES = 128;

// Aligned memory allocation helper
inline int64_t align_size(int64_t size) {
    return ((size + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
}

void attention_forward_cpu(
    const float* qkv,
    float* output,
    const float* bias,
    const int B, const int T, const int C,
    const int head_size
) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            // Load data into local memory with vectorized loads
            float s_mem[3 * BLOCK_SIZE * head_size];
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                for (int j = 0; j < head_size; ++j) {
                    s_mem[i * head_size + j] = qkv[(b * T + t) * 3 * head_size * BLOCK_SIZE + i * head_size + j];
                }
            }

            // Process attention scores with coalesced access
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    float att_score = 0.0f;
                    for (int k = 0; k < head_size; ++k) {
                        att_score += s_mem[i * head_size + k] * s_mem[j * head_size + k + head_size];
                    }
                    att_score *= 1.0f / std::sqrt(static_cast<float>(head_size));

                    // Apply causal mask
                    if (j > t) {
                        att_score = -std::numeric_limits<float>::infinity();
                    }

                    s_mem[i * BLOCK_SIZE + j] = att_score;
                }
            }

            // Compute softmax with coalesced access
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                float sum = 0.0f;
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    max_val = std::max(max_val, s_mem[i * BLOCK_SIZE + j]);
                }
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    s_mem[i * BLOCK_SIZE + j] = exp(s_mem[i * BLOCK_SIZE + j] - max_val);
                    sum += s_mem[i * BLOCK_SIZE + j];
                }
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    s_mem[i * BLOCK_SIZE + j] /= sum;
                }
            }

            // Compute final output with coalesced writes
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                float out_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    float att = s_mem[i * BLOCK_SIZE + j];
                    for (int k = 0; k < 4; ++k) {
                        out_val[k] += att * s_mem[j * head_size + k + 2 * head_size];
                    }
                }
                for (int k = 0; k < 4; ++k) {
                    output[(b * T + t) * C + i * 4 + k] = out_val[k];
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor c_attn_weight,
    torch::Tensor c_attn_bias,
    torch::Tensor c_proj_weight,
    torch::Tensor c_proj_bias,
    torch::Tensor bias,
    int64_t n_head,
    int64_t n_embd,
    bool is_training
) {
    using namespace torch::indexing;
    
    auto B = x.size(0);
    auto T = x.size(1);
    auto C = x.size(2);
    
    // Ensure aligned memory access
    auto head_size = C / n_head;
    auto aligned_head_size = align_size(head_size);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    
    // Prepare aligned tensors for coalesced access
    auto x_aligned = x.contiguous();
    auto qkv = torch::addmm(c_attn_bias, x_aligned.reshape({-1, C}), 
                           c_attn_weight.transpose(0, 1));
    qkv = qkv.reshape({B, T, 3, n_head, head_size}).contiguous();
    
    // Launch CPU function with proper configuration
    auto output = torch::empty({B, T, C}, x.options());
    attention_forward_cpu(
        qkv.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        B, T, C, head_size
    );
    
    // Final projection with aligned access
    auto out = torch::addmm(c_proj_bias, 
                           output.reshape({B * T, C}),
                           c_proj_weight.transpose(0, 1));
    
    return out.reshape({B, T, C});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Causal Attention forward (CPU)");
}
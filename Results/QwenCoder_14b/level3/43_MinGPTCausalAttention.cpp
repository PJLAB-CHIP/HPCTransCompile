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
    const float4* qkv,
    float4* output,
    const float* bias,
    const int B, const int T, const int C,
    const int head_size
) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int bid = b * T + t;
            int seq_offset = t * BLOCK_SIZE;
            
            // Load data into shared memory with vectorized loads
            float s_mem[3 * BLOCK_SIZE * head_size];
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                s_mem[i] = qkv[bid * BLOCK_SIZE + i].x;
                s_mem[BLOCK_SIZE + i] = qkv[bid * BLOCK_SIZE + i].y;
                s_mem[2 * BLOCK_SIZE + i] = qkv[bid * BLOCK_SIZE + i].z;
                s_mem[3 * BLOCK_SIZE + i] = qkv[bid * BLOCK_SIZE + i].w;
            }
            
            // Process attention scores with coalesced access
            for (int i = 0; i < BLOCK_SIZE; i += WARP_SIZE) {
                for (int row = 0; row < head_size; ++row) {
                    for (int col = 0; col < BLOCK_SIZE; ++col) {
                        const int global_col = seq_offset + col;
                        float att_score = 0.0f;
                        for (int k = 0; k < head_size; k += 4) {
                            float4 q_vec = {s_mem[row + k], s_mem[row + k + BLOCK_SIZE], s_mem[row + k + 2 * BLOCK_SIZE], s_mem[row + k + 3 * BLOCK_SIZE]};
                            float4 k_vec = {s_mem[col + k + head_size], s_mem[col + k + head_size + BLOCK_SIZE], s_mem[col + k + head_size + 2 * BLOCK_SIZE], s_mem[col + k + head_size + 3 * BLOCK_SIZE]};
                            att_score += q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                        }
                        att_score *= 1.0f / std::sqrt(static_cast<float>(head_size));
                        
                        // Apply causal mask
                        if (global_col > seq_offset + row) {
                            att_score = -std::numeric_limits<float>::infinity();
                        }
                        
                        // Store in shared memory with coalesced pattern
                        s_mem[row * BLOCK_SIZE + col] = att_score;
                    }
                }
            }
            
            // Compute softmax with coalesced access
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                float sum = 0.0f;
                
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    float val = s_mem[i * BLOCK_SIZE + j];
                    max_val = std::max(max_val, val);
                }
                
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    float val = exp(s_mem[i * BLOCK_SIZE + j] - max_val);
                    s_mem[i * BLOCK_SIZE + j] = val;
                    sum += val;
                }
                
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    s_mem[i * BLOCK_SIZE + j] /= sum;
                }
            }
            
            // Compute final output with coalesced writes
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                float4 out_val = {0.0f, 0.0f, 0.0f, 0.0f};
                const float4* v_ptr = reinterpret_cast<const float4*>(s_mem + 2 * head_size * BLOCK_SIZE);
                
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    float att = s_mem[i * BLOCK_SIZE + j];
                    float4 v_val = v_ptr[j];
                    out_val.x += att * v_val.x;
                    out_val.y += att * v_val.y;
                    out_val.z += att * v_val.z;
                    out_val.w += att * v_val.w;
                }
                
                output[bid * BLOCK_SIZE + i] = out_val;
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
    
    // Launch CPU function with proper grid/block configuration
    auto output = torch::empty({B, T, C}, x.options());
    
    attention_forward_cpu(
        reinterpret_cast<float4*>(qkv.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
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
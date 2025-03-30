#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

// Constants for memory alignment and optimization
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int ALIGN_BYTES = 128;

// Constant memory for frequently accessed parameters
__constant__ int64_t d_n_head;
__constant__ int64_t d_n_embd;
__constant__ float d_scale;

// Aligned memory allocation helper
inline int64_t align_size(int64_t size) {
    return ((size + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
}

std::vector<float4> load_data(const torch::Tensor& tensor) {
    std::vector<float4> data(tensor.numel());
    auto accessor = tensor.accessor<float, 2>();
    for (size_t i = 0; i < tensor.size(0); ++i) {
        for (size_t j = 0; j < tensor.size(1); ++j) {
            data[i * tensor.size(1) + j] = make_float4(accessor[i][j * 4 + 0], accessor[i][j * 4 + 1], accessor[i][j * 4 + 2], accessor[i][j * 4 + 3]);
        }
    }
    return data;
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
    
    // Copy constants to device
    cudaMemcpyToSymbol(d_n_head, &n_head, sizeof(int64_t));
    cudaMemcpyToSymbol(d_n_embd, &n_embd, sizeof(int64_t));
    cudaMemcpyToSymbol(d_scale, &scale, sizeof(float));
    
    // Prepare aligned tensors for coalesced access
    auto x_aligned = x.contiguous();
    auto qkv = torch::addmm(c_attn_bias, x_aligned.reshape({-1, C}), 
                           c_attn_weight.transpose(0, 1));
    qkv = qkv.reshape({B, T, 3, n_head, head_size}).contiguous();
    
    // Launch kernel with proper grid/block configuration
    dim3 grid(B * T / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    size_t shared_mem_size = 3 * aligned_head_size * BLOCK_SIZE * sizeof(float);
    
    auto output = torch::empty({B, T, C}, x.options());
    
    auto qkv_data = load_data(qkv);
    auto output_data = load_data(output);
    
    #pragma omp parallel for
    for (int64_t bid = 0; bid < B * T; ++bid) {
        int64_t batch_idx = bid / T;
        int64_t seq_offset = (bid % T) * BLOCK_SIZE;
        int64_t tid = omp_get_thread_num();
        
        // Load data into shared memory with vectorized loads
        std::vector<float> s_mem(BLOCK_SIZE * aligned_head_size * 3, 0.0f);
        for (int64_t i = 0; i < BLOCK_SIZE && tid + i < qkv_data.size(); ++i) {
            int64_t idx = bid * BLOCK_SIZE + tid + i;
            s_mem[tid * BLOCK_SIZE + i] = qkv_data[idx].x;
            s_mem[BLOCK_SIZE * aligned_head_size + tid * BLOCK_SIZE + i] = qkv_data[idx].y;
            s_mem[2 * BLOCK_SIZE * aligned_head_size + tid * BLOCK_SIZE + i] = qkv_data[idx].z;
        }
        
        // Process attention scores with coalesced access
        for (int64_t i = 0; i < BLOCK_SIZE; i += WARP_SIZE) {
            int64_t row = i / WARP_SIZE;
            int64_t col = i % WARP_SIZE;
            
            if (row < head_size && col < BLOCK_SIZE) {
                int64_t global_col = seq_offset + col;
                // Ensure coalesced access pattern for attention computation
                float att_score = 0.0f;
                for (int64_t k = 0; k < head_size; k += 4) {
                    float4 q_vec = {s_mem[row * BLOCK_SIZE + tid + k], s_mem[BLOCK_SIZE * aligned_head_size + row * BLOCK_SIZE + tid + k], s_mem[2 * BLOCK_SIZE * aligned_head_size + row * BLOCK_SIZE + tid + k], 0.0f};
                    float4 k_vec = {s_mem[col * BLOCK_SIZE + tid + k + head_size], s_mem[BLOCK_SIZE * aligned_head_size + col * BLOCK_SIZE + tid + k + head_size], s_mem[2 * BLOCK_SIZE * aligned_head_size + col * BLOCK_SIZE + tid + k + head_size], 0.0f};
                    att_score += q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z;
                }
                att_score *= d_scale;
                
                // Apply causal mask
                if (global_col > seq_offset + row) {
                    att_score = -std::numeric_limits<float>::infinity();
                }
                
                // Store in shared memory with coalesced pattern
                s_mem[row * BLOCK_SIZE + col] = att_score;
            }
        }
        
        // Compute softmax with coalesced access
        for (int64_t i = 0; i < BLOCK_SIZE; i++) {
            float max_val = -std::numeric_limits<float>::infinity();
            float sum = 0.0f;
            
            for (int64_t j = 0; j < BLOCK_SIZE; j++) {
                float val = s_mem[tid * BLOCK_SIZE + i + j * BLOCK_SIZE];
                max_val = std::max(max_val, val);
            }
            
            for (int64_t j = 0; j < BLOCK_SIZE; j++) {
                float val = std::exp(s_mem[tid * BLOCK_SIZE + i + j * BLOCK_SIZE] - max_val);
                s_mem[tid * BLOCK_SIZE + i + j * BLOCK_SIZE] = val;
                sum += val;
            }
            
            for (int64_t j = 0; j < BLOCK_SIZE; j++) {
                s_mem[tid * BLOCK_SIZE + i + j * BLOCK_SIZE] /= sum;
            }
        }
        
        // Compute final output with coalesced writes
        for (int64_t i = 0; i < BLOCK_SIZE; i++) {
            float4 out_val = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int64_t j = 0; j < BLOCK_SIZE; j++) {
                float att = s_mem[tid * BLOCK_SIZE + i + j * BLOCK_SIZE];
                float4 v_val = {s_mem[BLOCK_SIZE * aligned_head_size + tid * BLOCK_SIZE + i + j * BLOCK_SIZE], s_mem[BLOCK_SIZE * aligned_head_size + tid * BLOCK_SIZE + i + j * BLOCK_SIZE], s_mem[2 * BLOCK_SIZE * aligned_head_size + tid * BLOCK_SIZE + i + j * BLOCK_SIZE], 0.0f};
                out_val.x += att * v_val.x;
                out_val.y += att * v_val.y;
                out_val.z += att * v_val.z;
            }
            output_data[bid * BLOCK_SIZE + tid + i] = out_val;
        }
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Causal Attention forward (CPU)");
}
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>

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

__global__ void attention_forward_kernel(
    const float4* __restrict__ qkv,
    float4* __restrict__ output,
    const float* __restrict__ bias,
    const int B, const int T, const int C,
    const int head_size
) {
    extern __shared__ float s_mem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch_idx = bid / (T / BLOCK_SIZE);
    const int seq_offset = (bid % (T / BLOCK_SIZE)) * BLOCK_SIZE;
    
    // Load data into shared memory with vectorized loads
    if (tid < BLOCK_SIZE) {
        float4* s_qkv = reinterpret_cast<float4*>(s_mem);
        s_qkv[tid] = qkv[bid * BLOCK_SIZE + tid];
    }
    __syncthreads();
    
    // Process attention scores with coalesced access
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i += WARP_SIZE) {
        const int row = tid / WARP_SIZE;
        const int col = tid % WARP_SIZE;
        
        if (row < head_size && col < BLOCK_SIZE) {
            const int global_col = seq_offset + col;
            // Ensure coalesced access pattern for attention computation
            float att_score = 0.0f;
            #pragma unroll
            for (int k = 0; k < head_size; k += 4) {
                float4 q_vec = reinterpret_cast<float4*>(s_mem)[row + k];
                float4 k_vec = reinterpret_cast<float4*>(s_mem)[col + k + head_size];
                att_score += q_vec.x * k_vec.x + q_vec.y * k_vec.y + 
                            q_vec.z * k_vec.z + q_vec.w * k_vec.w;
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
    __syncthreads();
    
    // Compute softmax with coalesced access
    if (tid < BLOCK_SIZE) {
        float max_val = -std::numeric_limits<float>::infinity();
        float sum = 0.0f;
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float val = s_mem[tid * BLOCK_SIZE + i];
            max_val = max(max_val, val);
        }
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float val = exp(s_mem[tid * BLOCK_SIZE + i] - max_val);
            s_mem[tid * BLOCK_SIZE + i] = val;
            sum += val;
        }
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            s_mem[tid * BLOCK_SIZE + i] /= sum;
        }
    }
    __syncthreads();
    
    // Compute final output with coalesced writes
    if (tid < BLOCK_SIZE) {
        float4 out_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4* v_ptr = reinterpret_cast<const float4*>(s_mem + 2 * head_size * BLOCK_SIZE);
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float att = s_mem[tid * BLOCK_SIZE + i];
            float4 v_val = v_ptr[i];
            out_val.x += att * v_val.x;
            out_val.y += att * v_val.y;
            out_val.z += att * v_val.z;
            out_val.w += att * v_val.w;
        }
        
        output[bid * BLOCK_SIZE + tid] = out_val;
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
    
    attention_forward_kernel<<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &forward, "Coalesced Causal Attention forward (CUDA)");
}
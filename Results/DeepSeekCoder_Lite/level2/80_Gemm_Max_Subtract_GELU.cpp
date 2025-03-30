#include <torch/extension.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_DIM 32  // Aligned with warp size

__inline float gelu(float x) {
    // Fast GELU approximation using CUDA intrinsics
    const float a = 0.797884560802865f;
    const float b = 0.044715f;
    float cdf = 0.5f * (1.0f + tanhf(a * (x + b * x * x * x)));
    return x * cdf;
}

// Warp-aligned GEMM kernel
void warp_aligned_gemm_kernel(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ y,
                              int batch, int in_features, int out_features) {
    // Align with warp size for better occupancy
    __shared__ float tile_x[TILE_DIM][TILE_DIM];
    __shared__ float tile_w[TILE_DIM][TILE_DIM];

    int warp_id = omp_get_thread_num() % (BLOCK_SIZE / WARP_SIZE);
    int lane_id = omp_get_thread_num() % WARP_SIZE;
    
    int row = omp_get_thread_num() / WARP_SIZE;
    int col = omp_get_thread_num() % (BLOCK_SIZE / WARP_SIZE) * WARP_SIZE + lane_id;
    
    float sum = 0.0f;

    // Process input in warp-aligned tiles
    for (int t = 0; t < (in_features + TILE_DIM - 1) / TILE_DIM; t++) {
        int tile_x_col = t * TILE_DIM + lane_id;
        int tile_w_row = t * TILE_DIM + warp_id;
        
        // Collaborative loading using all threads in warp
        if (row < batch && tile_x_col < in_features) {
            tile_x[warp_id][lane_id] = x[row * in_features + tile_x_col];
        }
        if (col < out_features && tile_w_row < in_features) {
            tile_w[warp_id][lane_id] = weight[col * in_features + tile_w_row];
        }
        
        __sync();

        // Compute partial products
        #pragma omp simd
        for (int k = 0; k < TILE_DIM; k++) {
            sum += tile_x[warp_id][k] * tile_w[k][lane_id];
        }
        
        __sync();
    }

    // Write result with uniform control flow
    if (row < batch && col < out_features) {
        y[row * out_features + col] = sum + bias[col];
    }
}

// Warp-synchronized max reduction kernel
void warp_reduce_max_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int rows, int cols, int reduce_dim) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    int tid = omp_get_thread_num();
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    float max_val = -FLT_MAX;
    
    if (reduce_dim == 0) {
        // Reduce along rows (batch dimension)
        int col = omp_get_thread_num() / WARP_SIZE;
        for (int row = 0; row < rows; row += BLOCK_SIZE) {
            if (row + tid < rows) {
                max_val = fmaxf(max_val, input[(row + tid) * cols + col]);
            }
        }
    } else {
        // Reduce along columns (feature dimension)
        int row = omp_get_thread_num() / BLOCK_SIZE;
        for (int col = tid; col < cols; col += BLOCK_SIZE) {
            max_val = fmaxf(max_val, input[row * cols + col]);
        }
    }
    
    shared_data[tid] = max_val;
    __sync();
    
    // Warp-synchronized reduction
    if (tid < WARP_SIZE) {
        #pragma omp simd
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        
        if (lane_id == 0) {
            if (reduce_dim == 0) {
                output[omp_get_thread_num() * WARP_SIZE + warp_id] = max_val;
            } else if (warp_id == 0) {
                output[omp_get_thread_num() / BLOCK_SIZE] = max_val;
            }
        }
    }
}

// Fused mean-subtract-GELU kernel with warp-level operations
void warp_fused_mean_gelu_kernel(float* __restrict__ data,
                                 int rows, int cols) {
    __shared__ float warp_sums[WARP_SIZE];
    
    int row = omp_get_thread_num() / BLOCK_SIZE;
    int tid = omp_get_thread_num();
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    float sum = 0.0f;
    
    // Compute sum using warp-level reduction
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        sum += data[row * cols + col];
    }
    
    // Warp-synchronized reduction for mean computation
    #pragma omp simd
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __sync();
    
    // Final reduction and mean computation
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE; i++) {
            total_sum += warp_sums[i];
        }
        warp_sums[0] = total_sum / cols;  // Store mean
    }
    __sync();
    
    // Apply mean subtraction and GELU with minimal divergence
    float mean = warp_sums[0];
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float val = data[row * cols + col] - mean;
        data[row * cols + col] = gelu(val);
    }
}

torch::Tensor forward(torch::Tensor x, int max_dim, torch::Tensor weight, torch::Tensor bias) {
    int batch = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto y = torch::empty({batch, out_features}, x.options());
    
    // Launch warp-aligned GEMM
    #pragma omp parallel
    {
        warp_aligned_gemm_kernel(x.data_ptr<float>(), weight.data_ptr<float>(),
                                  bias.data_ptr<float>(), y.data_ptr<float>(),
                                  batch, in_features, out_features);
    }

    // Perform max reduction
    auto max_out = (max_dim == 0) ?
        torch::empty({1, out_features}, y.options()) :
        torch::empty({batch, 1}, y.options());
    
    int rows = (max_dim == 0) ? batch : 1;
    int cols = (max_dim == 0) ? out_features : batch;
    
    #pragma omp parallel
    {
        warp_reduce_max_kernel(y.data_ptr<float>(), max_out.data_ptr<float>(),
                                rows, cols, max_dim);
    }

    // Apply fused mean-subtract-GELU
    int final_rows = max_out.size(0);
    int final_cols = max_out.size(1);
    
    #pragma omp parallel
    {
        warp_fused_mean_gelu_kernel(max_out.data_ptr<float>(), final_rows, final_cols);
    }

    return max_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-aligned CPU forward implementation");
}
#include <torch/extension.h>
#include <vector>
#include <omp.h>
#include <cmath>
#include <algorithm>

typedef float4 vec4;

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double bn_eps,
    double bn_momentum,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor scale,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias
) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = gemm_bias.size(0);
    
    auto output = torch::empty({M, N}, x.options());
    
    const int num_threads = omp_get_max_threads();
    std::vector<float> s_max(num_threads * 32, -INFINITY);
    std::vector<float> s_sum(num_threads * 32, 0.0f);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int m = omp_get_thread_num() * (M / num_threads);
        
        for (int k = tid; k < K; k += num_threads) {
            float result = gemm_bias[k];
            
            for (int i = m; i < m + (M / num_threads); ++i) {
                const float* x_vec = &x[i * K + k];
                const float* weight_vec = &gemm_weight[k];
                result += x_vec[0] * weight_vec[0] + x_vec[1] * weight_vec[1] + x_vec[2] * weight_vec[2] + x_vec[3] * weight_vec[3];
            }
            
            float normalized = (result - running_mean[k]) * rsqrt(running_var[k] + bn_eps);
            normalized = normalized * bn_weight[k] + bn_bias[k];
            float scaled = normalized * scale[k];
            
            float max_val = scaled;
            
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = std::max(max_val, __shfl_down(max_val, offset));
                max_val = other;
            }
            
            if (tid % 32 == 0) {
                s_max[tid * 32 + (tid / 32)] = max_val;
            }
            __syncthreads();
            
            if (tid < 32) {
                float block_max = (tid < num_threads) ? s_max[tid * 32 + (tid / 32)] : -INFINITY;
                for (int offset = 16; offset > 0; offset /= 2) {
                    float other = std::max(block_max, __shfl_down(block_max, offset));
                    block_max = other;
                }
                s_max[tid * 32 + (tid / 32)] = block_max;
            }
            __syncthreads();
            
            float exp_val = expf(scaled - s_max[tid * 32 + (tid / 32)]);
            
            float local_sum = exp_val;
            for (int offset = 16; offset > 0; offset /= 2) {
                local_sum += __shfl_down(local_sum, offset);
            }
            
            if (tid % 32 == 0) {
                s_sum[tid * 32 + (tid / 32)] = local_sum;
            }
            __syncthreads();
            
            if (tid < 32) {
                float block_sum = (tid < num_threads) ? s_sum[tid * 32 + (tid / 32)] : 0.0f;
                for (int offset = 16; offset > 0; offset /= 2) {
                    block_sum += __shfl_down(block_sum, offset);
                }
                s_sum[tid * 32 + (tid / 32)] = block_sum;
            }
            __syncthreads();
            
            output[m * N + tid] = exp_val / s_sum[tid * 32 + (tid / 32)];
        }
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed fused GEMM+BN+Softmax CPU");
}
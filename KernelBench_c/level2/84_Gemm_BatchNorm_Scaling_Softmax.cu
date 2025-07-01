#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

typedef float4 vec4;

__global__ void fused_kernel_streamed(
    const float* __restrict__ x,
    const float* __restrict__ gemm_weight,
    const float* __restrict__ gemm_bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float bn_eps,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int m = blockIdx.x;
    
    float local_sum = 0.0f;
    __shared__ float s_max[64];
    __shared__ float s_sum[32];
    
    float result = gemm_bias[tid];
    
    const vec4* x_vec = (const vec4*)(&x[m * K]);
    const vec4* weight_vec = (const vec4*)(&gemm_weight[tid * K]);
    
    #pragma unroll 4
    for (int k = tid; k < K/4; k += blockDim.x) {
        vec4 x_data = x_vec[k];
        vec4 w_data = weight_vec[k];
        result += x_data.x * w_data.x + x_data.y * w_data.y + 
                 x_data.z * w_data.z + x_data.w * w_data.w;
    }
    
    for (int k = (K/4)*4 + tid; k < K; k += blockDim.x) {
        result += x[m * K + k] * gemm_weight[tid * K + k];
    }
    
    float normalized = (result - running_mean[tid]) * 
                      rsqrtf(running_var[tid] + bn_eps);
    normalized = normalized * bn_weight[tid] + bn_bias[tid];
    
    float scaled = normalized * scale[tid];
    
    float max_val = scaled;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = max(max_val, other);
    }
    
    if (tid % 32 == 0) {
        s_max[tid/32] = max_val;
    }
    __syncthreads();
    
    if (tid < 32) {
        float block_max = (tid < blockDim.x/32) ? s_max[tid] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, block_max, offset);
            block_max = max(block_max, other);
        }
        s_max[0] = block_max;
    }
    __syncthreads();
    
    float exp_val = expf(scaled - s_max[0]);
    
    local_sum = exp_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (tid % 32 == 0) {
        s_sum[tid/32] = local_sum;
    }
    __syncthreads();
    
    if (tid < 32) {
        float block_sum = (tid < blockDim.x/32) ? s_sum[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        s_sum[0] = block_sum;
    }
    __syncthreads();
    
    output[m * N + tid] = exp_val / s_sum[0];
}

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
    
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int rows_per_stream = (M + num_streams - 1) / num_streams;
    dim3 threads(N);
    
    for (int i = 0; i < num_streams; i++) {
        int start_row = i * rows_per_stream;
        int end_row = min(start_row + rows_per_stream, M);
        if (start_row >= M) break;
        
        dim3 blocks(end_row - start_row);
        
        fused_kernel_streamed<<<blocks, threads, 0, streams[i]>>>(x.data_ptr<float>() + start_row * K,
            gemm_weight.data_ptr<float>(),
            gemm_bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            static_cast<float>(bn_eps),
            bn_weight.data_ptr<float>(),
            bn_bias.data_ptr<float>(),
            scale.data_ptr<float>(),
            output.data_ptr<float>() + start_row * N,
            end_row - start_row, K, N);
    }
    
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed fused GEMM+BN+Softmax CUDA");
}
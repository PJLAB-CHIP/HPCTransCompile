#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define KERNEL_SIZE 3
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)

__device__ __forceinline__ float gelu_activate(float x) {
    return 0.5f * x * (1.f + erff(x / 1.41421356f));
}

__device__ __forceinline__ void warp_reduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void fused_conv_gelu_pool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w
) {
    extern __shared__ float shared_mem[];
    
    // Partition shared memory
    float* conv_weights = shared_mem;
    float* partial_sums = &shared_mem[in_channels * KERNEL_SIZE * KERNEL_SIZE];
    
    const int tid = threadIdx.x;
    const int n = blockIdx.y;
    const int c_out = blockIdx.x;
    
    // Load convolution weights into shared memory
    const int weight_size = in_channels * KERNEL_SIZE * KERNEL_SIZE;
    for (int i = tid; i < weight_size; i += BLOCK_SIZE) {
        conv_weights[i] = weight[c_out * weight_size + i];
    }
    __syncthreads();
    
    // Initialize partial sum
    float thread_sum = 0.0f;
    
    // Calculate number of pixels per thread
    const int total_pixels = out_h * out_w;
    const int pixels_per_thread = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Process pixels
    #pragma unroll 4
    for (int p = 0; p < pixels_per_thread; p++) {
        const int pixel_idx = tid + p * BLOCK_SIZE;
        if (pixel_idx < total_pixels) {
            const int out_row = pixel_idx / out_w;
            const int out_col = pixel_idx % out_w;
            
            float conv_result = 0.0f;
            
            #pragma unroll
            for (int ic = 0; ic < in_channels; ic++) {
                const float* in_ptr = &input[((n * in_channels + ic) * in_h + out_row) * in_w + out_col];
                const float* w_ptr = &conv_weights[ic * KERNEL_SIZE * KERNEL_SIZE];
                
                #pragma unroll
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        conv_result += in_ptr[kh * in_w + kw] * w_ptr[kh * KERNEL_SIZE + kw];
                    }
                }
            }
            
            // Add bias and apply GELU
            conv_result = gelu_activate(conv_result + bias[c_out]);
            thread_sum += conv_result;
        }
    }
    
    // Store partial sum in shared memory
    partial_sums[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within block using shared memory
    for (int s = BLOCK_SIZE/2; s > 32; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }
    
    // Final reduction within first warp
    if (tid < 32) {
        volatile float* smem = partial_sums;
        if (BLOCK_SIZE >= 64) smem[tid] += smem[tid + 32];
        warp_reduce(smem, tid);
        
        // Write result
        if (tid == 0) {
            output[n * out_channels + c_out] = smem[0] / float(total_pixels);
        }
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    
    const int N = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_channels = conv_weight.size(0);
    const int out_h = in_h - 2;
    const int out_w = in_w - 2;
    
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({N, out_channels}, options);
    
    dim3 grid(out_channels, N);
    
    // Calculate shared memory size: space for weights + space for partial sums
    const size_t shared_mem_size = 
        (in_channels * KERNEL_SIZE * KERNEL_SIZE + BLOCK_SIZE) * sizeof(float);
    
    fused_conv_gelu_pool_kernel<<<grid, BLOCK_SIZE, shared_mem_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, in_channels, in_h, in_w,
        out_channels, out_h, out_w
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized reduction Conv2d + GELU + GlobalAvgPool");
}
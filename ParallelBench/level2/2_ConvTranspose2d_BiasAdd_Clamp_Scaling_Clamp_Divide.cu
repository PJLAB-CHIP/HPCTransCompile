#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#define BLOCK_SIZE 256

// This kernel caches the bias values in shared memory to reduce redundant global memory accesses.
// Only one __syncthreads() is used after loading the bias values to ensure consistency, avoiding excessive synchronization.
// The kernel then processes the output tensor in a vectorized manner using float4 loads/stores and __ldg() for read-only access.

__global__ void shared_mem_post_process_kernel(
    float* __restrict__ output,
    const int total_size,    // total number of floats in the output
    const int height,
    const int width,
    const int channels,
    const float scaling_factor,
    const float* __restrict__ global_bias
) {
    // Allocate shared memory for the bias values
    extern __shared__ float s_bias[];

    // Each thread loads part of the bias array from global memory into shared memory
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        s_bias[i] = global_bias[i];
    }
    // Synchronize threads to ensure all bias values are loaded
    __syncthreads();

    // Compute the number of vectorized (float4) groups and the remaining elements
    int vec_size = total_size / 4;
    int remainder = total_size % 4;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float4* out_vec = reinterpret_cast<float4*>(output);
    int hw_size = height * width;

    // Process vectorized elements in groups of 4 using a grid-stride loop
    for (int i = tid; i < vec_size; i += stride) {
        // Load 4 floats at once using __ldg for enhanced read-only performance
        float4 data = __ldg(&out_vec[i]);
        int base_index = i * 4;
        float results[4];
        
        // Process each element of the float4
        {
            int index = base_index;
            int c = (index / hw_size) % channels;
            float val = data.x + s_bias[c];
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            val = val * scaling_factor;
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            results[0] = val / scaling_factor;
        }
        {
            int index = base_index + 1;
            int c = (index / hw_size) % channels;
            float val = data.y + s_bias[c];
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            val = val * scaling_factor;
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            results[1] = val / scaling_factor;
        }
        {
            int index = base_index + 2;
            int c = (index / hw_size) % channels;
            float val = data.z + s_bias[c];
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            val = val * scaling_factor;
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            results[2] = val / scaling_factor;
        }
        {
            int index = base_index + 3;
            int c = (index / hw_size) % channels;
            float val = data.w + s_bias[c];
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            val = val * scaling_factor;
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            results[3] = val / scaling_factor;
        }

        // Write the processed values back in a vectorized manner
        float4 out_val = make_float4(results[0], results[1], results[2], results[3]);
        out_vec[i] = out_val;
    }

    // Process any remaining elements that weren't covered by the vectorized loop
    int rem_start = vec_size * 4;
    for (int i = tid; i < remainder; i += stride) {
        int index = rem_start + i;
        int c = (index / hw_size) % channels;
        float val = __ldg(&output[index]) + s_bias[c];
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val * scaling_factor;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        output[index] = val / scaling_factor;
    }
}

// Forward function performs conv_transpose2d followed by the post-processing kernel

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    float scaling_factor,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    // Perform transposed convolution using PyTorch's built-in function
    auto output = torch::conv_transpose2d(
        x, conv_transpose, conv_transpose_bias,
        stride, padding, output_padding
    );

    const int batch_size = output.size(0);
    const int channels = output.size(1);
    const int height = output.size(2);
    const int width = output.size(3);
    const int total_size = batch_size * channels * height * width;

    int threads = BLOCK_SIZE;
    int vec_size = total_size / 4;  // Number of float4 groups
    int blocks = (vec_size + threads - 1) / threads;
    if (blocks == 0) blocks = 1;

    // Allocate shared memory for the bias: one float per channel
    size_t shared_mem_size = channels * sizeof(float);

    shared_mem_post_process_kernel<<<blocks, threads, shared_mem_size>>>(
        output.data_ptr<float>(),
        total_size,
        height,
        width,
        channels,
        scaling_factor,
        bias.data_ptr<float>()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared-memory bias optimized post-processing kernel with minimal synchronization (CUDA)");
}

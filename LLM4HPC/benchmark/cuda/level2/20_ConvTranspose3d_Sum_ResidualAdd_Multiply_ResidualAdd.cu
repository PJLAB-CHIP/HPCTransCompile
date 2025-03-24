#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel improves memory coalescing by processing data using vectorized loads/stores (float4).
// It also uses a grid-stride loop so that each thread processes multiple contiguous elements.
// The bias values are cached in shared memory to reduce redundant global memory accesses.
// The kernel computes: output[i] = conv_output[i] * (2.0f * conv_output[i] + bias[c] + 1.0f), where c = ((i / spatial_size) % channels).

__global__ void coalesced_vectorized_fused_operations_kernel(
    const float* __restrict__ conv_output,
    const float* __restrict__ element_bias,
    float* output,
    int num_elements,
    int channels,
    int spatial_size
) {
    // Allocate shared memory for bias values
    extern __shared__ float shared_bias[];

    // Each thread loads part of the bias into shared memory
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        shared_bias[i] = element_bias[i];
    }
    __syncthreads();

    // Process elements in vectorized manner using float4
    int total_vec = num_elements / 4;  // number of complete float4 groups
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop for the vectorized portion
    for (int i = idx; i < total_vec; i += blockDim.x * gridDim.x) {
        // Load 4 contiguous floats at once
        float4 in_vec = reinterpret_cast<const float4*>(conv_output)[i];
        int base = i * 4;
        float4 out_vec;
        // Unroll the computation for the 4 elements
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int global_idx = base + j;
            int c = (global_idx / spatial_size) % channels;
            // Access the j-th component of the vector
            float original = ((float*)&in_vec)[j];
            float b = shared_bias[c];
            ((float*)&out_vec)[j] = original * (2.0f * original + b + 1.0f);
        }
        // Store the computed 4 elements back to global memory
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Process any remaining elements that don't form a complete float4
    int remainder = num_elements % 4;
    int start = total_vec * 4;
    for (int i = idx; i < remainder; i += blockDim.x * gridDim.x) {
        int global_idx = start + i;
        int c = (global_idx / spatial_size) % channels;
        float orig = conv_output[global_idx];
        output[global_idx] = orig * (2.0f * orig + shared_bias[c] + 1.0f);
    }
}

// The forward function applies the standard conv_transpose3d and then launches the optimized kernel.

torch::Tensor forward(
    torch::Tensor x,
    int stride,
    int padding,
    int output_padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    // Compute the transposed convolution using PyTorch's optimized function
    auto conv_result = torch::conv_transpose3d(
        x,
        conv_transpose,
        conv_transpose_bias,
        stride,
        padding,
        output_padding
    );

    // Get dimensions; assume conv_result is in shape [N, C, D, H, W] and is contiguous
    auto sizes = conv_result.sizes();
    int channels = sizes[1];
    int spatial_size = sizes[2] * sizes[3] * sizes[4];  // D * H * W
    int num_elements = conv_result.numel();

    // Prepare the output tensor
    auto output = torch::empty_like(conv_result);

    // Configure kernel launch parameters
    const int threads_per_block = 256;
    int total_vec = num_elements / 4;
    int blocks = (total_vec > 0) ? ((total_vec + threads_per_block - 1) / threads_per_block) : ((num_elements + threads_per_block - 1) / threads_per_block);

    // Launch the kernel with dynamic shared memory allocation for bias
    coalesced_vectorized_fused_operations_kernel<<<blocks, threads_per_block, channels * sizeof(float)>>>(
        conv_result.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        channels,
        spatial_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Vectorized Fused ConvTranspose3D Kernel with Channel-wise Bias");
}

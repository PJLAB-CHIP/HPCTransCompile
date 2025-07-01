#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float4 load_float4(const float* addr) {
    float4 val;
    val = *reinterpret_cast<const float4*>(addr);
    return val;
}

__device__ __forceinline__ void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

__device__ __forceinline__ float process_value(float val, const float* bias, int bias_idx) {
    // ReLU
    val = fmaxf(0.0f, val);
    
    // LeakyReLU
    val = fmaxf(0.01f * val, val);
    
    // GELU
    const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
    val = 0.5f * val * (1.0f + tanhf(sqrt_2_over_pi * (val + 0.044715f * powf(val, 3.0f))));
    
    // Sigmoid
    val = 1.0f / (1.0f + expf(-val));
    
    // Add bias
    val += __ldg(&bias[bias_idx]);
    
    return val;
}

__global__ void apply_activations_and_bias_kernel(
    float* __restrict__ output, const float* __restrict__ bias,
    int batch_size, int out_channels, int depth, int height, int width
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int block_offset = blockIdx.x * BLOCK_SIZE;
    
    // Calculate spatial dimensions for coalesced access
    const int spatial_size = depth * height * width;
    const int elements_per_channel = spatial_size;
    
    // Process 4 elements at a time when possible
    const int vector_idx = (block_offset + tid) * 4;
    const int total_elements = batch_size * out_channels * spatial_size;
    
    if (vector_idx < total_elements - 3) {
        // Load 4 consecutive elements
        float4 data = load_float4(&output[vector_idx]);
        
        // Calculate bias index for the current position
        int base_idx = vector_idx / spatial_size;
        int bias_idx = base_idx % out_channels;
        
        // Process each component
        data.x = process_value(data.x, bias, bias_idx);
        data.y = process_value(data.y, bias, bias_idx);
        data.z = process_value(data.z, bias, bias_idx);
        data.w = process_value(data.w, bias, bias_idx);
        
        // Store results back
        store_float4(&output[vector_idx], data);
    }
    // Handle remaining elements
    else if (vector_idx < total_elements) {
        for (int i = 0; i < 4 && vector_idx + i < total_elements; ++i) {
            int curr_idx = vector_idx + i;
            float val = output[curr_idx];
            int bias_idx = (curr_idx / spatial_size) % out_channels;
            output[curr_idx] = process_value(val, bias, bias_idx);
        }
    }
}

torch::Tensor module_fn_cuda(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(conv_weight);
    CHECK_INPUT(conv_bias);
    CHECK_INPUT(bias);

    auto output = torch::conv3d(x, conv_weight, conv_bias);

    int batch_size = output.size(0);
    int out_channels = output.size(1);
    int depth = output.size(2);
    int height = output.size(3);
    int width = output.size(4);

    int total_vectors = (batch_size * out_channels * depth * height * width + 3) / 4;
    int blocks = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;

    apply_activations_and_bias_kernel<<<blocks, BLOCK_SIZE>>>(
        output.data_ptr<float>(), bias.data_ptr<float>(),
        batch_size, out_channels, depth, height, width
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cuda, "CUDA implementation of module_fn");
}
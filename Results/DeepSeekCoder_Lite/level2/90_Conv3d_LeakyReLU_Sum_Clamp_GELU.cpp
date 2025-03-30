#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <omp.h>

// This function processes the main part of the tensor in groups of 4 elements using 128-bit aligned loads/stores via float4.
void my_kernel_vectorized_cpu(
    const float* __restrict__ input,
    const float* __restrict__ sum_tensor,
    float* __restrict__ output,
    const int64_t num_vectorized,
    const int64_t width,
    const int64_t height,
    const int64_t depth,
    const int64_t channels) {

    #pragma omp parallel for
    for (int64_t id = 0; id < num_vectorized; ++id) {
        int64_t base = id * 4;
        // Cast input to float4 pointer and use __restrict__ for read-only access
        const float4* input4 = reinterpret_cast<const float4*>(input);
        float4 in_val = input4[id];
        float4 res;

        // Process each element in the vector
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int64_t idx = base + i;
            // Compute the 5D tensor indices given the flattened index idx
            int64_t w = idx % width;
            int64_t h = (idx / width) % height;
            int64_t d = (idx / (width * height)) % depth;
            int64_t c = (idx / (width * height * depth)) % channels;
            
            float x;
            if (i == 0) x = in_val.x;
            else if (i == 1) x = in_val.y;
            else if (i == 2) x = in_val.z;
            else x = in_val.w;
            
            // Use branchless LeakyReLU
            float y = fmaxf(x, 0.2f * x);
            // Add bias from sum_tensor using __restrict__ for read-only access
            y += sum_tensor[c];
            // Clamp the value to [-1, 1]
            y = fmaxf(fminf(y, 1.0f), -1.0f);
            // Apply GELU activation
            float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (y + 0.044715f * y * y * y)));
            y = y * cdf;
            
            // Assign the computed value to the corresponding component
            if (i == 0) res.x = y;
            else if (i == 1) res.y = y;
            else if (i == 2) res.z = y;
            else res.w = y;
        }
        
        // Write back the result using a 128-bit aligned store
        float4* output4 = reinterpret_cast<float4*>(output);
        output4[id] = res;
    }
}

// This function processes any remaining elements that do not fit into a group of 4
void my_kernel_remainder_cpu(
    const float* __restrict__ input,
    const float* __restrict__ sum_tensor,
    float* __restrict__ output,
    const int64_t start,
    const int64_t num_elements,
    const int64_t width,
    const int64_t height,
    const int64_t depth,
    const int64_t channels) {
    
    #pragma omp parallel for
    for (int64_t idx = 0; idx < num_elements; ++idx) {
        int64_t global_idx = start + idx;
        if (global_idx < num_elements) {
            float x = __restrict__ input[global_idx];
            float y = fmaxf(x, 0.2f * x);
            int64_t w = global_idx % width;
            int64_t h = (global_idx / width) % height;
            int64_t d = (global_idx / (width * height)) % depth;
            int64_t c = (global_idx / (width * height * depth)) % channels;
            y += sum_tensor[c];
            y = fmaxf(fminf(y, 1.0f), -1.0f);
            float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (y + 0.044715f * y * y * y)));
            output[global_idx] = y * cdf;
        }
    }
}

// Launcher that selects between vectorized and remainder kernels
void my_kernel_launcher_cpu(
    torch::Tensor& x,
    torch::Tensor& sum_tensor) {
    
    const int64_t num_elements = x.numel();
    const int64_t channels = x.size(1);
    
    // Calculate the number of groups of 4 elements and remainder
    int64_t num_vectorized = num_elements / 4;
    int64_t remainder = num_elements % 4;
    
    // Launch the vectorized kernel with 128 threads per block for better occupancy
    int threads = 128;
    int blocks = (num_vectorized + threads - 1) / threads;
    my_kernel_vectorized_cpu(
        x.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        x.data_ptr<float>(),
        num_vectorized,
        x.size(4),
        x.size(3),
        x.size(2),
        channels
    );
    
    // Launch the remainder kernel if there are leftover elements
    if (remainder > 0) {
        // Use fixed block size of 128 threads for better efficiency
        int threads_rem = 128;
        int blocks_rem = (remainder + threads_rem - 1) / threads_rem;
        int64_t start = num_vectorized * 4;
        my_kernel_remainder_cpu(
            x.data_ptr<float>(),
            sum_tensor.data_ptr<float>(),
            x.data_ptr<float>(),
            start,
            num_elements,
            x.size(4),
            x.size(3),
            x.size(2),
            channels
        );
    }
}

// Forward function that performs the 3D convolution and applies the custom CPU kernel
torch::Tensor forward_cpu(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor sum_tensor) {

    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(conv_weight.is_cpu(), "conv_weight must be a CPU tensor");
    TORCH_CHECK(conv_bias.is_cpu(), "conv_bias must be a CPU tensor");
    TORCH_CHECK(sum_tensor.is_cpu(), "sum_tensor must be a CPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be of type float32");

    // Perform 3D convolution
    auto x_conv = at::conv3d(x, conv_weight, conv_bias);

    // Ensure output is contiguous
    auto output = x_conv.contiguous();

    // Apply the optimized kernel with aligned 128-bit memory accesses using __restrict__
    my_kernel_launcher_cpu(output, sum_tensor);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Custom forward function (CPU) with aligned 128-bit vectorized loads/stores");
}
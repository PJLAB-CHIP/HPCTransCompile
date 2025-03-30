#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <omp.h>

// Define block size for kernels
constexpr int BLOCK_SIZE = 256;

// Scalar kernel using __ldg() for read-only loads
void clamp_and_scale_scalar(const float* __restrict__ in, float* __restrict__ out, int num_elements, float factor, float min_val, float max_val) {
    #pragma omp parallel for
    for (int idx = 0; idx < num_elements; ++idx) {
        // Use __ldg() to load from global memory in read-only cache
        float v = in[idx];
        v = v * (2.0f * factor);
        v = std::fmin(std::fmax(v, min_val), max_val);
        out[idx] = v;
    }
}

// Vectorized kernel processing 4 floats at a time using float4
void clamp_and_scale_vectorized(const float4* __restrict__ in, float4* __restrict__ out, int num_elements4, float factor, float min_val, float max_val) {
    #pragma omp parallel for
    for (int idx = 0; idx < num_elements4; ++idx) {
        // Load a vector of 4 floats using __ldg()
        float4 v = in[idx];
        float s = 2.0f * factor;
        v.x = std::fmin(std::fmax(v.x * s, min_val), max_val);
        v.y = std::fmin(std::fmax(v.y * s, min_val), max_val);
        v.z = std::fmin(std::fmax(v.z * s, min_val), max_val);
        v.w = std::fmin(std::fmax(v.w * s, min_val), max_val);
        out[idx] = v;
    }
}

// Kernel to perform LogSumExp across rows and apply Mish activation
void logsumexp_mish_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols, int num_elements) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int rows_per_thread = (rows + num_threads - 1) / num_threads;
        int row_start = tid * rows_per_thread;
        int row_end = std::min(row_start + rows_per_thread, rows);

        for (int row = row_start; row < row_end; ++row) {
            float max_val = -INFINITY;
            for (int i = 0; i < cols; ++i) {
                float val = input[row * cols + i];
                max_val = std::fmax(max_val, val);
            }

            float sum = 0.0f;
            for (int i = 0; i < cols; ++i) {
                float v = input[row * cols + i];
                sum += expf(v - max_val);
            }

            float lse = logf(sum) + max_val;
            float softplus = log1pf(expf(lse));
            float mish = lse * tanhf(softplus);
            output[row] = lse * mish;
        }
    }
}

// Forward function that implements the complete fused operation
torch::Tensor module_fn_forward(
    torch::Tensor x,
    float scale_factor,
    float clamp_min,
    float clamp_max,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // Ensure inputs are contiguous for aligned memory accesses
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // 1. Matrix multiplication and bias addition
    auto out = torch::mm(x, weight.transpose(0, 1));
    out.add_(bias);

    // 2. Fuse scaling, residual addition, and clamping using a custom kernel
    int num_elements = out.numel();
    // Check for 128-bit alignment and divisibility by 4 for vectorized operations
    bool use_vectorized = (num_elements % 4 == 0) && (((uintptr_t)out.data_ptr<float>()) % 16 == 0);

    if (use_vectorized) {
        int num_elements4 = num_elements / 4;
        clamp_and_scale_vectorized(reinterpret_cast<const float4*>(out.data_ptr<float>()), reinterpret_cast<float4*>(out.data_ptr<float>()), num_elements4, scale_factor, clamp_min, clamp_max);
    } else {
        clamp_and_scale_scalar(out.data_ptr<float>(), out.data_ptr<float>(), num_elements, scale_factor, clamp_min, clamp_max);
    }
    
    // 3. Apply LogSumExp and Mish activation along rows using a reduction kernel
    auto output = torch::empty({out.size(0), 1}, out.options());
    logsumexp_mish_kernel(out.data_ptr<float>(), output.data_ptr<float>(), out.size(0), out.size(1));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Forward pass for module_fn (CPU)");
}
#include <torch/extension.h>
#include <omp.h>
#include <cmath>

// Device helper functions with forced inlining
__forceinline__ float apply_ops(float val, float add_value, float multiply_value) {
    val = val + add_value;
    val = std::fminf(val, 0.0f);
    float t = std::tanh(0.79788456f * (val + 0.044715f * val * val * val));
    return (val * 0.5f * (1.0f + t)) * multiply_value;
}

__forceinline__ double apply_ops(double val, double add_value, double multiply_value) {
    val = val + add_value;
    val = (val < 0.0) ? val : 0.0;
    double t = std::tanh(0.79788456 * (val + 0.044715 * val * val * val));
    return (val * 0.5 * (1.0 + t)) * multiply_value;
}

// Warp-optimized vectorized kernel for float using float4 for coalesced access
void vectorized_float_kernel(float* __restrict__ x, int64_t num_vecs, int64_t size, float add_val, float mult_val) {
    int total_threads = omp_get_max_threads();
    for (int vec_idx = 0; vec_idx < num_vecs; ++vec_idx) {
        int idx = vec_idx * 4;
        float4 v = { x[idx], x[idx + 1], x[idx + 2], x[idx + 3] };
        v.x = apply_ops(v.x, add_val, mult_val);
        v.y = apply_ops(v.y, add_val, mult_val);
        v.z = apply_ops(v.z, add_val, mult_val);
        v.w = apply_ops(v.w, add_val, mult_val);
        x[idx] = v.x;
        x[idx + 1] = v.y;
        x[idx + 2] = v.z;
        x[idx + 3] = v.w;
    }

    // Tail processing: Only the first thread handles the remainder
    if (omp_get_thread_num() == 0) {
        int tail_offset = num_vecs * 4;
        int tail_elems = size - tail_offset;  // Number of remaining elements (< 4)
        for (int i = 0; i < tail_elems; ++i) {
            x[tail_offset + i] = apply_ops(x[tail_offset + i], add_val, mult_val);
        }
    }
}

// Warp-optimized vectorized kernel for double using double2 for coalesced access
void vectorized_double_kernel(double* __restrict__ x, int64_t num_vecs, int64_t size, double add_val, double mult_val) {
    int total_threads = omp_get_max_threads();
    for (int vec_idx = 0; vec_idx < num_vecs; ++vec_idx) {
        int idx = vec_idx * 2;
        double2 v = { x[idx], x[idx + 1] };
        v.x = apply_ops(v.x, add_val, mult_val);
        v.y = apply_ops(v.y, add_val, mult_val);
        x[idx] = v.x;
        x[idx + 1] = v.y;
    }

    // Tail processing: Only the first thread handles the remainder
    if (omp_get_thread_num() == 0) {
        int tail_offset = num_vecs * 2;
        int tail_elems = size - tail_offset;  // Remaining elements (< 2)
        for (int i = 0; i < tail_elems; ++i) {
            x[tail_offset + i] = apply_ops(x[tail_offset + i], add_val, mult_val);
        }
    }
}

// CPU launcher for the elementwise operations
torch::Tensor elementwise_cpu(
    torch::Tensor x,
    double add_value,
    double multiply_value
) {
    // Ensure tensor is contiguous
    x = x.contiguous();
    int64_t numel = x.numel();
    const int threads = 256;

    if (x.scalar_type() == at::ScalarType::Float) {
        int64_t num_vecs = numel / 4;
        int blocks = (num_vecs + threads - 1) / threads;
        #pragma omp parallel for
        for (int i = 0; i < blocks; ++i) {
            vectorized_float_kernel(x.data_ptr<float>(), num_vecs, numel, static_cast<float>(add_value), static_cast<float>(multiply_value));
        }
    } else if (x.scalar_type() == at::ScalarType::Double) {
        int64_t num_vecs = numel / 2;
        int blocks = (num_vecs + threads - 1) / threads;
        #pragma omp parallel for
        for (int i = 0; i < blocks; ++i) {
            vectorized_double_kernel(x.data_ptr<double>(), num_vecs, numel, add_value, multiply_value);
        }
    }

    return x;
}

// Main function: applies conv_transpose2d then elementwise operations
torch::Tensor module_fn(
    torch::Tensor x,
    int64_t stride,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    double add_value,
    double multiply_value
) {
    if (!x.is_cuda() || !conv_transpose.is_cuda() || !conv_transpose_bias.is_cuda()) {
        throw std::runtime_error("All input tensors must be CUDA tensors");
    }

    // Apply transposed convolution
    x = at::conv_transpose2d(x, conv_transpose, conv_transpose_bias, {stride});
    // Apply elementwise operations using our CPU kernel
    x = elementwise_cpu(x, add_value, multiply_value);

    return x;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Module function forward");
}
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device helper functions with forced inlining
__device__ __forceinline__ float apply_ops(float val, float add_value, float multiply_value) {
    val = val + add_value;
    val = fminf(val, 0.0f);
    float t = tanhf(0.79788456f * (val + 0.044715f * val * val * val));
    return (val * 0.5f * (1.0f + t)) * multiply_value;
}

__device__ __forceinline__ double apply_ops(double val, double add_value, double multiply_value) {
    val = val + add_value;
    val = (val < 0.0) ? val : 0.0;
    double t = tanh(0.79788456 * (val + 0.044715 * val * val * val));
    return (val * 0.5 * (1.0 + t)) * multiply_value;
}

// Warp-optimized vectorized kernel for float using float4 for coalesced access
// Tail elements are handled by the first warp using __shfl_sync to broadcast the tail count
__global__ void vectorized_float_kernel(float* __restrict__ x, int64_t num_vecs, int64_t size, float add_val, float mult_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float4* x_vec = reinterpret_cast<float4*>(x);

    // Process main vectorized portion in a grid-stride loop
    for (int i = tid; i < num_vecs; i += total_threads) {
        float4 v = x_vec[i];
        v.x = apply_ops(v.x, add_val, mult_val);
        v.y = apply_ops(v.y, add_val, mult_val);
        v.z = apply_ops(v.z, add_val, mult_val);
        v.w = apply_ops(v.w, add_val, mult_val);
        x_vec[i] = v;
    }

    // Tail processing: Only the first block handles the remainder using warp-level primitives
    if (blockIdx.x == 0) {
        int tail_offset = num_vecs * 4;
        int tail_elems = size - tail_offset;  // Number of remaining elements (< 4)
        // Broadcast tail_elems from lane 0 across the first warp
        int valid_tail = __shfl_sync(0xffffffff, tail_elems, 0);
        // Only threads in the first warp (threadIdx.x < 32) participate
        int lane = threadIdx.x;
        if (lane < valid_tail) {
            int idx = tail_offset + lane;
            x[idx] = apply_ops(x[idx], add_val, mult_val);
        }
    }
}

// Warp-optimized vectorized kernel for double using double2 for coalesced access
// Tail processing is similarly handled using __shfl_sync
__global__ void vectorized_double_kernel(double* __restrict__ x, int64_t num_vecs, int64_t size, double add_val, double mult_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    double2* x_vec = reinterpret_cast<double2*>(x);

    for (int i = tid; i < num_vecs; i += total_threads) {
        double2 v = x_vec[i];
        v.x = apply_ops(v.x, add_val, mult_val);
        v.y = apply_ops(v.y, add_val, mult_val);
        x_vec[i] = v;
    }

    if (blockIdx.x == 0) {
        int tail_offset = num_vecs * 2;
        int tail_elems = size - tail_offset;  // Remaining elements (< 2)
        int valid_tail = __shfl_sync(0xffffffff, tail_elems, 0);
        int lane = threadIdx.x;
        if (lane < valid_tail) {
            int idx = tail_offset + lane;
            x[idx] = apply_ops(x[idx], add_val, mult_val);
        }
    }
}

// CUDA launcher for the elementwise operations
torch::Tensor elementwise_cuda(
    torch::Tensor x,
    double add_value,
    double multiply_value
) {
    // Ensure tensor is contiguous
    x = x.contiguous();
    auto numel = x.numel();
    const int threads = 256;
    
    if (x.scalar_type() == at::ScalarType::Float) {
        // Process in chunks of 4 floats
        int64_t num_vecs = numel / 4;
        int blocks = (num_vecs + threads - 1) / threads;
        vectorized_float_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            num_vecs,
            numel,
            static_cast<float>(add_value),
            static_cast<float>(multiply_value)
        );
    } else if (x.scalar_type() == at::ScalarType::Double) {
        // Process in chunks of 2 doubles
        int64_t num_vecs = numel / 2;
        int blocks = (num_vecs + threads - 1) / threads;
        vectorized_double_kernel<<<blocks, threads>>>(
            x.data_ptr<double>(),
            num_vecs,
            numel,
            add_value,
            multiply_value
        );
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
    // Apply elementwise operations using our CUDA kernel
    x = elementwise_cuda(x, add_value, multiply_value);

    return x;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Module function forward");
}

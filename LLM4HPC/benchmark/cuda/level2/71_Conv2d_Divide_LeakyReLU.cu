#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define a vectorized type for coalesced memory access with alignment for 4 consecutive elements
template <typename scalar_t>
struct alignas(sizeof(scalar_t) * 4) Vec4 {
    scalar_t v[4];
};

// CUDA kernel with even workload distribution among threads
// Each thread calculates its start and count for both the vectorized part and tail part to ensure balanced work

template <typename scalar_t>
__global__ void even_div_leaky_relu_kernel(
    scalar_t* __restrict__ x,
    scalar_t divisor,
    scalar_t negative_slope,
    int64_t size
) {
    // Total number of threads in the grid
    int totalThreads = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Process vectorized loads (groups of 4 elements)
    int nVec = size / 4; // number of complete Vec4 groups
    int remainder = size % 4; // leftover elements

    // Evenly distribute nVec among threads
    int baseVec = nVec / totalThreads;
    int extraVec = nVec % totalThreads;
    int startVec = (tid < extraVec) ? tid * (baseVec + 1) : tid * baseVec + extraVec;
    int countVec = (tid < extraVec) ? (baseVec + 1) : baseVec;

    Vec4<scalar_t>* vec_ptr = reinterpret_cast<Vec4<scalar_t>*>(x);
    for (int i = startVec; i < startVec + countVec; i++) {
        Vec4<scalar_t> data = vec_ptr[i];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            scalar_t cached_divisor = 1.0 / divisor; scalar_t val = data.v[j] * cached_divisor;
            data.v[j] = (val >= static_cast<scalar_t>(0)) ? val : val * negative_slope;
        }
        vec_ptr[i] = data;
    }

    // Process the tail elements that don't fit into a complete Vec4
    int tailOffset = nVec * 4;
    int baseTail = remainder / totalThreads;
    int extraTail = remainder % totalThreads;
    int startTail = (tid < extraTail) ? tailOffset + tid * (baseTail + 1) : tailOffset + tid * baseTail + extraTail;
    int countTail = (tid < extraTail) ? (baseTail + 1) : baseTail;

    for (int i = 0; i < countTail; i++) {
        int idx = startTail + i;
        // Check bounds safety
        if (idx < size) {
            scalar_t val = x[idx] / divisor;
            x[idx] = (val >= static_cast<scalar_t>(0)) ? val : val * negative_slope;
        }
    }
}

// CPU implementation for completeness
template <typename scalar_t>
void even_div_leaky_relu_cpu_impl(
    scalar_t* x, scalar_t divisor, scalar_t negative_slope, int64_t size
) {
    for (int64_t i = 0; i < size; i++) {
        scalar_t val = x[i] / divisor;
        x[i] = (val >= static_cast<scalar_t>(0)) ? val : val * negative_slope;
    }
}

// Dispatcher function: selects the CUDA or CPU implementation

torch::Tensor even_div_leaky_relu(torch::Tensor x, double divisor, double negative_slope) {
    x = x.contiguous();
    const int64_t size = x.numel();

    if (x.is_cuda()) {
        // Choose a block configuration that covers the vectorized part
        int threads = 1024;
        int nVec = size / 4;
        // Ensure we have enough threads but not more than necessary
        int blocks = (nVec + threads - 1) / threads;
        if(blocks == 0) blocks = 1; // safety

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "even_div_leaky_relu_cuda", ([&] {
            scalar_t divisor_val = static_cast<scalar_t>(divisor);
            scalar_t negative_slope_val = static_cast<scalar_t>(negative_slope);
            even_div_leaky_relu_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                divisor_val,
                negative_slope_val,
                size
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "even_div_leaky_relu_cpu", ([&] {
            scalar_t divisor_val = static_cast<scalar_t>(divisor);
            scalar_t negative_slope_val = static_cast<scalar_t>(negative_slope);
            even_div_leaky_relu_cpu_impl<scalar_t>(
                x.data_ptr<scalar_t>(),
                divisor_val,
                negative_slope_val,
                size
            );
        }));
    }

    return x;
}

// Forward function: applies convolution followed by the evenly distributed division and LeakyReLU operation

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    double divisor
) {
    // Convolution using PyTorch's ATen conv2d
    x = at::conv2d(x, conv_weight, conv_bias);
    // Apply element-wise division and LeakyReLU with negative_slope = 0.01
    x = even_div_leaky_relu(x, divisor, 0.01);
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Convolution, division, and LeakyReLU forward (even load distribution)");
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <type_traits>

// This kernel is optimized to reduce global memory latency by using __ldg() for read-only accesses
// and by aligning memory accesses to 128-bit boundaries via vectorized loads (float4 for float, double2 for double).
// It computes a fused linear transform followed by instance normalization, residual addition, and multiplication.


template <typename T>
__global__ void fused_linear_instancenorm_ldg_kernel(
    const T* __restrict__ input,      // [batch_size, in_features]
    const T* __restrict__ residual,   // [batch_size, out_features]
    const T* __restrict__ weight,     // [out_features, in_features]
    const T* __restrict__ bias,       // [out_features]
    T* __restrict__ output,           // [batch_size, out_features]
    const int batch_size,
    const int in_features,
    const int out_features,
    const float eps
) {
    // Each block processes one batch sample
    int batch_idx = blockIdx.x;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    // Allocate shared memory:
    // s_linear: to store final linear layer output per instance [out_features]
    // s_scratch: scratch space for dot-product reduction [blockDim.x * blockDim.y]
    // s_reduction: scratch space for mean/variance reduction [blockDim.x]
    extern __shared__ char shared_mem[];
    T* s_linear   = reinterpret_cast<T*>(shared_mem);
    T* s_scratch  = s_linear + out_features;          // size: blockDim.x * blockDim.y
    T* s_reduction = s_scratch + (blockDim.x * blockDim.y); // size: blockDim.x

    // Step 1: Compute the linear layer output with optimized global loads using __ldg() and 128-bit aligned accesses.
    // Each thread with index (tid_x, tid_y) processes a subset of in_features for a given output feature index.
    for (int out_idx = tid_x; out_idx < out_features; out_idx += blockDim.x) {
        T partial = static_cast<T>(0);
        int offset_input  = batch_idx * in_features;
        int offset_weight = out_idx * in_features;
        
        // Set vectorization parameters based on type:
        // For float: use vec_size = 4 (i.e. float4 loads, 16 bytes = 128 bits).
        // For double: use vec_size = 2 (i.e. double2 loads, 16 bytes).
        constexpr int vec_size = (std::is_same<T, float>::value) ? 4 : (std::is_same<T, double>::value ? 2 : 1);

        int aligned_bound = (in_features / vec_size) * vec_size;

        if (vec_size > 1) {
            if constexpr (std::is_same<T, float>::value) {
                const float4* input_vec  = reinterpret_cast<const float4*>(input + offset_input);
                const float4* weight_vec = reinterpret_cast<const float4*>(weight + offset_weight);
                int vec_count = aligned_bound / 4;
                for (int i = tid_y; i < vec_count; i += blockDim.y) {
                    // Use __ldg() for read-only load
                    float4 a = __ldg(input_vec + i);
                    float4 b = __ldg(weight_vec + i);
                    partial += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
            } else if constexpr (std::is_same<T, double>::value) {
                const double2* input_vec  = reinterpret_cast<const double2*>(input + offset_input);
                const double2* weight_vec = reinterpret_cast<const double2*>(weight + offset_weight);
                int vec_count = aligned_bound / 2;
                for (int i = tid_y; i < vec_count; i += blockDim.y) {
                    double2 a = __ldg(input_vec + i);
                    double2 b = __ldg(weight_vec + i);
                    partial += a.x * b.x + a.y * b.y;
                }
            }
            // Process any remaining elements
            for (int i = aligned_bound + tid_y; i < in_features; i += blockDim.y) {
                partial += __ldg(input + offset_input + i) * __ldg(weight + offset_weight + i);
            }
        } else {
            for (int i = tid_y; i < in_features; i += blockDim.y) {
                partial += __ldg(input + offset_input + i) * __ldg(weight + offset_weight + i);
            }
        }

        // Store the partial dot-product result in shared scratch memory
        int index = tid_x * blockDim.y + tid_y;
        s_scratch[index] = partial;
    }
    __syncthreads();

    // Step 2: Reduce the partial sums along threadIdx.y for each output feature
    for (int out_idx = tid_x; out_idx < out_features; out_idx += blockDim.x) {
        if (tid_y == 0) {
            T sum_val = s_scratch[out_idx * blockDim.y];
            #pragma unroll
            for (int k = 1; k < blockDim.y; k++) {
                sum_val += s_scratch[out_idx * blockDim.y + k];
            }
            // Add bias term using __ldg()
            s_linear[out_idx] = sum_val + __ldg(bias + out_idx);
        }
    }
    __syncthreads();

    // Step 3: Compute the mean of the linear outputs
    T mean_partial = static_cast<T>(0);
    for (int i = tid_x; i < out_features; i += blockDim.x) {
        mean_partial += s_linear[i];
    }
    if (tid_y == 0) {
        s_reduction[tid_x] = mean_partial;
    }
    __syncthreads();

    if (tid_y == 0) {
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid_x < stride) {
                s_reduction[tid_x] += s_reduction[tid_x + stride];
            }
            __syncthreads();
        }
        s_reduction[0] = s_reduction[0] / out_features;
    }
    __syncthreads();
    T mean = s_reduction[0];

    // Step 4: Compute the variance
    T var_partial = static_cast<T>(0);
    for (int i = tid_x; i < out_features; i += blockDim.x) {
        T diff = s_linear[i] - mean;
        var_partial += diff * diff;
    }
    if (tid_y == 0) {
        s_reduction[tid_x] = var_partial;
    }
    __syncthreads();
    if (tid_y == 0) {
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid_x < stride) {
                s_reduction[tid_x] += s_reduction[tid_x + stride];
            }
            __syncthreads();
        }
        s_reduction[0] = s_reduction[0] / out_features;
    }
    __syncthreads();
    T var = s_reduction[0];
    T inv_std = rsqrtf(var + eps);

    // Step 5: Normalize the linear output and apply residual addition and multiplication
    int batch_offset = batch_idx * out_features;
    for (int i = tid_x; i < out_features; i += blockDim.x) {
        T norm_val = (s_linear[i] - mean) * inv_std;
        T res_val = __ldg(residual + batch_offset + i);
        output[batch_offset + i] = (norm_val + res_val) * res_val;
    }
}


// Host function to launch the kernel

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor y,
    float eps,
    float momentum,  // For API compatibility
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = y.size(1);

    auto output = torch::empty_like(y);

    // Configure block and grid dimensions
    const int block_x = 128;
    const int block_y = 4;
    dim3 block(block_x, block_y);
    dim3 grid(batch_size);

    // Allocate shared memory: s_linear (out_features) + s_scratch (block_x * block_y) + s_reduction (block_x)
    size_t shared_mem_size = sizeof(at::Half) * 0;  // dummy initialization
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_linear_instancenorm_ldg_kernel", ([&] {
        shared_mem_size = sizeof(scalar_t) * (out_features + block_x * block_y + block_x);
        fused_linear_instancenorm_ldg_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused linear, instance norm, residual add and multiply with __ldg() and 128-bit aligned loads");
}

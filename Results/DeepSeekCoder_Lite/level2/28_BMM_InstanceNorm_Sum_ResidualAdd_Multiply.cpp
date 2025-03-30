#include <torch/extension.h>
#include <omp.h>
#include <type_traits>
#include <cmath>

// This function is optimized to reduce global memory latency by using __ldg() for read-only accesses
// and by aligning memory accesses to 128-bit boundaries via vectorized loads (float4 for float, double2 for double).
// It computes a fused linear transform followed by instance normalization, residual addition, and multiplication.

template <typename T>
void fused_linear_instancenorm_ldg_kernel(
    const T* __restrict__ input,      // [batch_size, in_features]
    const T* __restrict__ residual,   // [batch_size, out_features]
    const T* __restrict__ weight,     // [out_features, in_features]
    const T* __restrict__ bias,       // [out_features]
    T* __restrict__ output,           // [batch_size, out_features]
    const int batch_size,
    const int in_features,
    const int out_features,
    const float eps,
    const int block_x,
    const int block_y
) {
    // Each thread processes a subset of out_features
    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        // Allocate shared memory:
        // s_linear: to store final linear layer output per instance [out_features]
        // s_scratch: scratch space for dot-product reduction [block_x * block_y]
        // s_reduction: scratch space for mean/variance reduction [block_x]
        std::vector<T> s_linear(out_features);
        std::vector<T> s_scratch(block_x * block_y);
        std::vector<T> s_reduction(block_x);

        // Step 1: Compute the linear layer output with optimized global loads using __ldg() and 128-bit aligned accesses.
        for (int out_idx = omp_get_thread_num(); out_idx < out_features; out_idx += omp_get_num_threads()) {
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
                    for (int i = 0; i < vec_count; ++i) {
                        // Use __ldg() for read-only load
                        float4 a = __ldg(input_vec + i);
                        float4 b = __ldg(weight_vec + i);
                        partial += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                    }
                } else if constexpr (std::is_same<T, double>::value) {
                    const double2* input_vec  = reinterpret_cast<const double2*>(input + offset_input);
                    const double2* weight_vec = reinterpret_cast<const double2*>(weight + offset_weight);
                    int vec_count = aligned_bound / 2;
                    for (int i = 0; i < vec_count; ++i) {
                        double2 a = __ldg(input_vec + i);
                        double2 b = __ldg(weight_vec + i);
                        partial += a.x * b.x + a.y * b.y;
                    }
                }
                // Process any remaining elements
                for (int i = aligned_bound; i < in_features; ++i) {
                    partial += __ldg(input + offset_input + i) * __ldg(weight + offset_weight + i);
                }
            } else {
                for (int i = 0; i < in_features; ++i) {
                    partial += __ldg(input + offset_input + i) * __ldg(weight + offset_weight + i);
                }
            }

            // Store the partial dot-product result in shared scratch memory
            int index = omp_get_thread_num() * block_y + omp_get_thread_num();
            s_scratch[index] = partial;
        }

        // Step 2: Reduce the partial sums along threadIdx.y for each output feature
        for (int out_idx = omp_get_thread_num(); out_idx < out_features; out_idx += omp_get_num_threads()) {
            T sum_val = s_scratch[out_idx * block_y];
            for (int k = 1; k < block_y; ++k) {
                sum_val += s_scratch[out_idx * block_y + k];
            }
            // Add bias term using __ldg()
            s_linear[out_idx] = sum_val + __ldg(bias + out_idx);
        }

        // Step 3: Compute the mean of the linear outputs
        T mean_partial = static_cast<T>(0);
        for (int i = omp_get_thread_num(); i < out_features; i += omp_get_num_threads()) {
            mean_partial += s_linear[i];
        }
        s_reduction[omp_get_thread_num()] = mean_partial;

        // Reduce mean_partial
        for (int stride = omp_get_num_threads() / 2; stride > 0; stride /= 2) {
            if (omp_get_thread_num() < stride) {
                s_reduction[omp_get_thread_num()] += s_reduction[omp_get_thread_num() + stride];
            }
            #pragma omp barrier
        }
        T mean = s_reduction[0] / out_features;

        // Step 4: Compute the variance
        T var_partial = static_cast<T>(0);
        for (int i = omp_get_thread_num(); i < out_features; i += omp_get_num_threads()) {
            T diff = s_linear[i] - mean;
            var_partial += diff * diff;
        }
        s_reduction[omp_get_thread_num()] = var_partial;

        // Reduce var_partial
        for (int stride = omp_get_num_threads() / 2; stride > 0; stride /= 2) {
            if (omp_get_thread_num() < stride) {
                s_reduction[omp_get_thread_num()] += s_reduction[omp_get_thread_num() + stride];
            }
            #pragma omp barrier
        }
        T var = s_reduction[0] / out_features;
        T inv_std = static_cast<T>(1.0) / sqrt(var + eps);

        // Step 5: Normalize the linear output and apply residual addition and multiplication
        int batch_offset = batch_idx * out_features;
        for (int i = omp_get_thread_num(); i < out_features; i += omp_get_num_threads()) {
            T norm_val = (s_linear[i] - mean) * inv_std;
            T res_val = __ldg(residual + batch_offset + i);
            output[batch_offset + i] = (norm_val + res_val) * res_val;
        }
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

    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_linear_instancenorm_ldg_kernel", ([&] {
        fused_linear_instancenorm_ldg_kernel<scalar_t>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features,
            eps,
            block_x,
            block_y
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused linear, instance norm, residual add and multiply with __ldg() and 128-bit aligned loads");
}
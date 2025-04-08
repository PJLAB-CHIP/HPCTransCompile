#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <immintrin.h> // For SIMD intrinsics

// Function to compute the fused linear transform, instance normalization, residual addition, and multiplication
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor y,
    float eps,
    float momentum,  // For API compatibility
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(!x.is_cuda(), "Input tensor must be on CPU device");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = y.size(1);

    auto output = torch::empty_like(y);

    // Use OpenMP for parallel processing
    #pragma omp parallel for collapse(2)
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        std::vector<float> s_linear(out_features, 0.0f);
        std::vector<float> s_scratch(block_x * block_y, 0.0f);
        std::vector<float> s_reduction(block_x, 0.0f);

        // Step 1: Compute the linear layer output with optimized global loads using SIMD intrinsics
        for (int out_idx = 0; out_idx < out_features; ++out_idx) {
            float partial = 0.0f;
            int offset_input = batch_idx * in_features;
            int offset_weight = out_idx * in_features;

            // Set vectorization parameters based on type:
            // For float: use vec_size = 4 (i.e. float4 loads, 16 bytes = 128 bits).
            constexpr int vec_size = 4;

            int aligned_bound = (in_features / vec_size) * vec_size;

            if (vec_size > 1) {
                const float* input_vec = x.data_ptr<float>() + offset_input;
                const float* weight_vec = weight.data_ptr<float>() + offset_weight;
                int vec_count = aligned_bound / vec_size;

                for (int i = 0; i < vec_count; i += block_y) {
                    __m256 a = _mm256_loadu_ps(input_vec + i);
                    __m256 b = _mm256_loadu_ps(weight_vec + i);
                    __m256 c = _mm256_mul_ps(a, b);
                    __m256 d = _mm256_hadd_ps(c, c);
                    __m256 e = _mm256_hadd_ps(d, d);
                    partial += e[0] + e[1];
                }

                // Process any remaining elements
                for (int i = aligned_bound; i < in_features; ++i) {
                    partial += x.data_ptr<float>()[offset_input + i] * weight.data_ptr<float>()[offset_weight + i];
                }
            } else {
                for (int i = 0; i < in_features; ++i) {
                    partial += x.data_ptr<float>()[offset_input + i] * weight.data_ptr<float>()[offset_weight + i];
                }
            }

            // Store the partial dot-product result in scratch memory
            s_scratch[out_idx * block_y] = partial;
        }

        // Step 2: Reduce the partial sums along threadIdx.y for each output feature
        for (int out_idx = 0; out_idx < out_features; ++out_idx) {
            float sum_val = s_scratch[out_idx * block_y];
            for (int k = 1; k < block_y; ++k) {
                sum_val += s_scratch[out_idx * block_y + k];
            }
            // Add bias term
            s_linear[out_idx] = sum_val + bias.data_ptr<float>()[out_idx];
        }

        // Step 3: Compute the mean of the linear outputs
        float mean_partial = 0.0f;
        for (int i = 0; i < out_features; ++i) {
            mean_partial += s_linear[i];
        }
        float mean = mean_partial / out_features;

        // Step 4: Compute the variance
        float var_partial = 0.0f;
        for (int i = 0; i < out_features; ++i) {
            float diff = s_linear[i] - mean;
            var_partial += diff * diff;
        }
        float var = var_partial / out_features;
        float inv_std = std::sqrt(1.0f / (var + eps));

        // Step 5: Normalize the linear output and apply residual addition and multiplication
        int batch_offset = batch_idx * out_features;
        for (int i = 0; i < out_features; ++i) {
            float norm_val = (s_linear[i] - mean) * inv_std;
            float res_val = y.data_ptr<float>()[batch_offset + i];
            output.data_ptr<float>()[batch_offset + i] = (norm_val + res_val) * res_val;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused linear, instance norm, residual add and multiply with SIMD intrinsics");
}

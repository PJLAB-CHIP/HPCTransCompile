#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <vector>

// Optimized warp-level reduction using shuffle intrinsics
__inline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 32/2; offset > 0; offset /= 2) {
        val += __builtin_shuffle_float(val, offset);
    }
    return val;
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int D = x.size(1);
    const int threads = std::min<int>(1024, ((D + 3)/4) * 4); // Align with vector loads

    #pragma omp parallel for
    for (int row = 0; row < N; ++row) {
        float sum = 0.0f;
        if (D >= 4) {
            const float4* x4 = reinterpret_cast<const float4*>(x.data_ptr<float>() + row * D);
            for (int col = 0; col < D/4; ++col) {
                float4 vals = x4[col];
                sum += std::fabs(vals.x) + std::fabs(vals.y) + std::fabs(vals.z) + std::fabs(vals.w);
            }
            // Handle remaining elements
            for (int col = D/4*4; col < D; ++col) {
                sum += std::fabs(x.data_ptr<float>()[row * D + col]);
            }
        } else {
            for (int col = 0; col < D; ++col) {
                sum += std::fabs(x.data_ptr<float>()[row * D + col]);
            }
        }

        // Warp-level reduction
        sum = warpReduceSum(sum);

        // Store warp results to shared memory
        if (row < 32) {
            sum = (sum == 0.0f) ? 1e-12f : sum;
            out.data_ptr<float>()[row] = sum;
        }

        // Normalize with vectorized stores when possible
        if (D >= 4) {
            float4* out4 = reinterpret_cast<float4*>(out.data_ptr<float>() + row * D);
            const float4* x4 = reinterpret_cast<const float4*>(x.data_ptr<float>() + row * D);
            for (int col = 0; col < D/4; ++col) {
                float4 vals = x4[col];
                vals.x /= sum;
                vals.y /= sum;
                vals.z /= sum;
                vals.w /= sum;
                out4[col] = vals;
            }
            // Handle remaining elements
            for (int col = D/4*4; col < D; ++col) {
                out.data_ptr<float>()[row * D + col] = x.data_ptr<float>()[row * D + col] / sum;
            }
        } else {
            for (int col = 0; col < D; ++col) {
                out.data_ptr<float>()[row * D + col] = x.data_ptr<float>()[row * D + col] / sum;
            }
        }
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA with hybrid optimizations)");
}
#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Function to perform warp-level reduction
float warpReduceSum(float val) {
    #pragma omp parallel for reduction(+:val)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += val;
    }
    return val;
}

void l1_norm_hybrid_cpu(const float* x, float* out, const int N, const int D) {
    #pragma omp parallel for
    for (int row = 0; row < N; ++row) {
        float sum = 0.0f;
        if (D >= 4) {
            const float4* x4 = reinterpret_cast<const float4*>(x + row * D);
            for (int col = 0; col < D/4; ++col) {
                float4 vals = x4[col];
                sum += fabsf(vals.x) + fabsf(vals.y) + fabsf(vals.z) + fabsf(vals.w);
            }
            for (int col = (D/4)*4; col < D; ++col) {
                sum += fabsf(x[row * D + col]);
            }
        } else {
            for (int col = 0; col < D; ++col) {
                sum += fabsf(x[row * D + col]);
            }
        }

        sum = warpReduceSum(sum);

        if (sum == 0.0f) {
            sum = 1e-12f;
        }

        if (D >= 4) {
            float4* out4 = reinterpret_cast<float4*>(out + row * D);
            const float4* x4 = reinterpret_cast<const float4*>(x + row * D);
            for (int col = 0; col < D/4; ++col) {
                float4 vals = x4[col];
                vals.x /= sum;
                vals.y /= sum;
                vals.z /= sum;
                vals.w /= sum;
                out4[col] = vals;
            }
            for (int col = (D/4)*4; col < D; ++col) {
                out[row * D + col] = x[row * D + col] / sum;
            }
        } else {
            for (int col = 0; col < D; ++col) {
                out[row * D + col] = x[row * D + col] / sum;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(!x.is_cuda(), "Input tensor must be on CPU.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int D = x.size(1);

    l1_norm_hybrid_cpu(x.data_ptr<float>(), out.data_ptr<float>(), N, D);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CPU with hybrid optimizations)");
}

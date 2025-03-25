#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <omp.h>

// Fused CPU kernel: performs division, 3D max pooling over non-overlapping windows,
// and then adaptive average pooling (summing over all pooled windows) with bias addition.
void fused_divide_maxpool_avg_cpu(const float* in,
                                   float* out,
                                   int N, int C,
                                   int D, int H, int W,
                                   int poolD, int poolH, int poolW,
                                   int OD, int OH, int OW,
                                   float divisor,
                                   const float* bias) {
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float partialSum = 0.0f;
            int total_windows = OD * OH * OW;

            for (int idx = 0; idx < total_windows; ++idx) {
                int ow = idx % OW;
                int tmp = idx / OW;
                int oh = tmp % OH;
                int od = tmp / OH;

                int d_start = od * poolD;
                int h_start = oh * poolH;
                int w_start = ow * poolW;

                float max_val = -FLT_MAX;
                for (int d = d_start; d < d_start + poolD; ++d) {
                    for (int h = h_start; h < h_start + poolH; ++h) {
                        for (int w = w_start; w < w_start + poolW; ++w) {
                            int index = (((n * C + c) * D + d) * H + h) * W + w;
                            float val = in[index] * (1.0f / divisor);
                            max_val = std::max(max_val, val);
                        }
                    }
                }
                partialSum += max_val;
            }

            float avg = partialSum / static_cast<float>(total_windows);
            out[n * C + c] = avg + bias[c];
        }
    }
}

// Reduction CPU kernel to sum the (N, C) tensor along a chosen dimension
void reduction_sum_cpu(const float* in,
                        float* out,
                        int N, int C, int sum_dim) {
    if (sum_dim == 1) {
        #pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                sum += in[n * C + c];
            }
            out[n] = sum;
        }
    } else if (sum_dim == 0) {
        #pragma omp parallel for
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                sum += in[n * C + c];
            }
            out[c] = sum;
        }
    }
}

// The forward_cpu function performs:
// 1) 3D convolution (using at::conv3d for correctness),
// 2) a fused CPU kernel that computes division, 3D max pooling across windows, adaptive average pooling, and bias addition,
// 3) a reduction CPU kernel to sum the (N, C) tensor along the specified dimension (sum_dim == 0 or 1).
torch::Tensor forward_cpu(torch::Tensor x,
                           double divisor,
                           std::vector<int64_t> pool_size,
                           int64_t sum_dim,
                           torch::Tensor conv_weight,
                           torch::Tensor conv_bias,
                           torch::Tensor bias) {
    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor.");
    TORCH_CHECK(!conv_weight.is_cuda(), "conv_weight must be a CPU tensor.");
    TORCH_CHECK(!conv_bias.is_cuda(), "conv_bias must be a CPU tensor.");
    TORCH_CHECK(!bias.is_cuda(), "bias must be a CPU tensor.");

    // 1) 3D convolution using PyTorch's conv3d
    auto conv_out = at::conv3d(x, conv_weight, conv_bias);
    // conv_out shape: (N, C, D, H, W)
    int N = conv_out.size(0);
    int C = conv_out.size(1);
    int D = conv_out.size(2);
    int H = conv_out.size(3);
    int W = conv_out.size(4);

    // Pooling window sizes
    int poolD = pool_size[0];
    int poolH = pool_size[1];
    int poolW = pool_size[2];

    // Compute output dimensions for the pooling stage (assumes perfect divisibility)
    int OD = D / poolD;
    int OH = H / poolH;
    int OW = W / poolW;

    auto options = conv_out.options();
    // Output of fused kernel: adaptive average pooling result per (n, c)
    auto avg_out = at::empty({N, C}, options);

    // Launch fused CPU kernel
    fused_divide_maxpool_avg_cpu(
        conv_out.data_ptr<float>(),
        avg_out.data_ptr<float>(),
        N, C, D, H, W,
        poolD, poolH, poolW,
        OD, OH, OW,
        static_cast<float>(divisor),
        bias.data_ptr<float>()
    );

    // 3) Reduction: sum over the (N, C) result along an input-specified dimension.
    torch::Tensor final_out;
    if (sum_dim == 1) {
        // Sum over channels; final output shape: (N)
        final_out = at::empty({N}, options);
        reduction_sum_cpu(
            avg_out.data_ptr<float>(),
            final_out.data_ptr<float>(),
            N, C, sum_dim
        );
    } else if (sum_dim == 0) {
        // Sum over batch; final output shape: (C)
        final_out = at::empty({C}, options);
        reduction_sum_cpu(
            avg_out.data_ptr<float>(),
            final_out.data_ptr<float>(),
            N, C, sum_dim
        );
    } else {
        TORCH_CHECK(false, "sum_dim must be 0 or 1");
    }

    return final_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Fused conv3d, divide, max pool, adaptive avg pool, bias add, and reduction kernel");
}
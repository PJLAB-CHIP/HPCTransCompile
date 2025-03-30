#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <omp.h>

// Fallback kernel for general stride (uses original global memory access)
void conv_transpose1d_kernel(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation)
{
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            for (int l_out = 0; l_out < L_out; ++l_out) {
                float value = (bias != nullptr) ? bias[c_out] : 0.0f;
                for (int c = 0; c < C_in; ++c) {
                    for (int k = 0; k < K_w; ++k) {
                        int l_in_nom = l_out * stride - padding + k * dilation;
                        if (l_in_nom >= 0 && l_in_nom < L_in) {
                            int l_in = l_in_nom;
                            value += x[n * C_in * L_in + c * L_in + l_in] * weight[c * C_out * K_w + c_out * K_w + k];
                        }
                    }
                }
                y[n * C_out * L_out + c_out * L_out + l_out] = value;
            }
        }
    }
}

// Optimized kernel for stride==1 using shared memory tiling to align accesses
void conv_transpose1d_kernel_tiled(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int padding, int dilation)
{
    int tile_size = 128;  // number of output elements processed per block (tunable)
    int num_tiles = (L_out + tile_size - 1) / tile_size;

    #pragma omp parallel for
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int l_out_start = tile_idx * tile_size;
        int l_out_end = std::min(l_out_start + tile_size, L_out);
        int num_threads = l_out_end - l_out_start;

        std::vector<float> shmem(C_in * (tile_size + (K_w - 1) * dilation) * num_threads, 0.0f);
        int load_start = l_out_start * stride - padding + (K_w - 1) * dilation;

        for (int c = 0; c < C_in; ++c) {
            for (int k = 0; k < K_w; ++k) {
                int l_global = load_start + k * dilation;
                for (int l_out = l_out_start; l_out < l_out_end; ++l_out) {
                    int l_in = l_global + l_out * stride;
                    if (l_in >= 0 && l_in < L_in) {
                        shmem[c * (tile_size + (K_w - 1) * dilation) + (l_out - l_out_start) + (K_w - 1) * dilation - k * dilation] = x[N * C_in * L_in + c * L_in + l_in];
                    }
                    l_global += stride;
                }
            }
        }

        for (int l_out = l_out_start; l_out < l_out_end; ++l_out) {
            float result = (bias != nullptr) ? bias[c_out] : 0.0f;
            for (int c = 0; c < C_in; ++c) {
                for (int k = 0; k < K_w; ++k) {
                    int offset = (K_w - 1) * dilation - k * dilation + (l_out - l_out_start);
                    result += shmem[c * (tile_size + (K_w - 1) * dilation) + offset] * weight[c * C_out * K_w + c_out * K_w + k];
                }
            }
            y[N * C_out * L_out + c_out * L_out + l_out] = result;
        }
    }
}

torch::Tensor conv_transpose1d_forward(
    const torch::Tensor& x,            // x: torch.Tensor
    const torch::Tensor& weight,       // weight: torch.Tensor
    const torch::Tensor& bias = torch::Tensor())
{
    // Convert inputs to contiguous tensors
    auto x_contiguous = x.contiguous();
    auto weight_contiguous = weight.contiguous();
    auto bias_contiguous = bias.contiguous();

    // Check if tensors are on the correct device
    TORCH_CHECK(x_contiguous.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight_contiguous.is_cuda(), "Weight tensor must be on CUDA device");
    TORCH_CHECK(bias_contiguous.is_cuda(), "Bias tensor must be on CUDA device");

    // Extract tensor dimensions
    int N = x_contiguous.size(0);
    int C_in = x_contiguous.size(1);
    int L_in = x_contiguous.size(2);
    int K_w = weight_contiguous.size(2);
    int C_out = weight_contiguous.size(1);

    // Compute output length
    int L_out = (L_in - 1) * 1 - 2 * 0 + (K_w - 1) * 1 + 1;

    // Allocate output tensor
    auto options = x.options();
    torch::Tensor y = torch::empty({N, C_out, L_out}, options);

    if (1 == 1) {
        // Use the optimized tiled kernel when stride==1
        int tile_size = 128;  // number of output elements processed per block (tunable)
        int num_tiles = (L_out + tile_size - 1) / tile_size;

        #pragma omp parallel for
        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            int l_out_start = tile_idx * tile_size;
            int l_out_end = std::min(l_out_start + tile_size, L_out);
            int num_threads = l_out_end - l_out_start;

            std::vector<float> shmem(C_in * (tile_size + (K_w - 1) * 1) * num_threads, 0.0f);
            int load_start = l_out_start * 1 - 0 + (K_w - 1) * 1;

            for (int c = 0; c < C_in; ++c) {
                for (int k = 0; k < K_w; ++k) {
                    int l_global = load_start + k * 1;
                    for (int l_out = l_out_start; l_out < l_out_end; ++l_out) {
                        int l_in = l_global + l_out * 1;
                        if (l_in >= 0 && l_in < L_in) {
                            shmem[c * (tile_size + (K_w - 1) * 1) + (l_out - l_out_start) + (K_w - 1) * 1 - k * 1] = x_contiguous[N * C_in * L_in + c * L_in + l_in];
                        }
                        l_global += 1;
                    }
                }
            }

            for (int l_out = l_out_start; l_out < l_out_end; ++l_out) {
                float result = (bias.numel() > 0) ? bias_contiguous[c_out] : 0.0f;
                for (int c = 0; c < C_in; ++c) {
                    for (int k = 0; k < K_w; ++k) {
                        int offset = (K_w - 1) * 1 - k * 1 + (l_out - l_out_start);
                        result += shmem[c * (tile_size + (K_w - 1) * 1) + offset] * weight_contiguous[c * C_out * K_w + c_out * K_w + k];
                    }
                }
                y[N * C_out * L_out + c_out * L_out + l_out] = result;
            }
        }
    } else {
        // Fallback to the original kernel for general stride
        conv_transpose1d_kernel(
            x_contiguous.data_ptr<float>(),
            weight_contiguous.data_ptr<float>(),
            bias_contiguous.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C_in, C_out, L_in, L_out, K_w,
            1, 0, 1);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv_transpose1d_forward,
        "Conv Transpose1D forward (CPU)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none()
    );
}
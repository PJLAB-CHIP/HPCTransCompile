#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <omp.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Warp-level reduction function
__forceinline__ float warp_reduce_sum(float val) {
    #pragma omp simd
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __builtin_shfl_down(val, offset);
    }
    return val;
}

__forceinline__ void warp_reduce(float& sum, float& sumsq) {
    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_transpose,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int64_t groups,
    double eps) {

    auto y = at::conv_transpose3d(
        x,
        conv_transpose,
        /*bias=*/c10::nullopt,
        /*stride=*/{1, 1, 1},
        /*padding=*/{0, 0, 0},
        /*output_padding=*/{0, 0, 0},
        /*groups=*/1,
        /*dilation=*/{1, 1, 1}
    );

    int N = y.size(0);
    int C = y.size(1);
    int D = y.size(2);
    int H = y.size(3);
    int W = y.size(4);
    int G = groups;

    float* y_data = y.data_ptr<float>();
    const float* gamma = group_norm_weight.data_ptr<float>();
    const float* beta = group_norm_bias.data_ptr<float>();

    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            int channels_per_group = C / G;
            int c_start = g * channels_per_group;
            int group_size = channels_per_group * D * H * W;
            int group_offset = n * C * D * H * W + c_start * D * H * W;

            float warp_sum = 0.0;
            float warp_sumsq = 0.0;

            #pragma omp simd
            for (int i = 0; i < group_size; i += BLOCK_SIZE) {
                int idx = group_offset + i + omp_get_thread_num() * BLOCK_SIZE;
                float val = y_data[idx];
                val = fmaxf(val, 0.0);
                y_data[idx] = val;
                warp_sum += val;
                warp_sumsq += val * val;
            }

            // Warp-level reduction
            warp_reduce(warp_sum, warp_sumsq);

            // Store warp results
            int wid = omp_get_thread_num() * NUM_WARPS + omp_get_thread_num() % WARP_SIZE;
            if (omp_get_thread_num() % NUM_WARPS == 0) {
                warp_sum = warp_sum / group_size;
                warp_sumsq = warp_sumsq / group_size - warp_sum * warp_sum + eps;
                float inv_std = rsqrtf(warp_sumsq + static_cast<float>(eps));

                #pragma omp simd
                for (int i = 0; i < group_size; i += BLOCK_SIZE) {
                    int idx = group_offset + i + omp_get_thread_num() * BLOCK_SIZE;
                    float val = y_data[idx];
                    int base_idx = idx * 4;
                    for (int j = 0; j < 4; ++j) {
                        int channel_idx = (base_idx + j) / (D * H * W);
                        int c = c_start + channel_idx;
                        y_data[idx] = (val - warp_sum) * inv_std * gamma[c] + beta[c];
                    }
                }
            }
        }
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ConvTranspose3D + ReLU + GroupNorm with optimized atomic operations (CPU)");
}
#include <torch/extension.h>
#include <cfloat>
#include <omp.h>

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias
) {
    x = x.contiguous();
    conv_weight = conv_weight.contiguous();
    conv_bias = conv_bias.contiguous();

    auto conv_output = at::conv3d(x, conv_weight, conv_bias, {1, 1, 1}, {0, 0, 0});
    auto softmax_output = at::softmax(conv_output, /*dim=*/1);

    const int N = softmax_output.size(0);
    const int C = softmax_output.size(1);
    const int D = softmax_output.size(2);
    const int H = softmax_output.size(3);
    const int W = softmax_output.size(4);

    const int outD = D / 4;
    const int outH = H / 4;
    const int outW = W / 4;

    auto options = softmax_output.options();
    auto output = torch::empty({N, C, outD, outH, outW}, options);

    // Calculate optimal items per block based on GPU characteristics
    const int items_per_block = 4;  // Process 4 output elements per block
    const int total_elements = N * C * outD * outH * outW;
    const int num_blocks = (total_elements + items_per_block - 1) / items_per_block;

    #pragma omp parallel for
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int item_idx = block_idx * items_per_block;
        if (item_idx >= total_elements) continue;

        const int n = item_idx / (C * outD * outH * outW);
        int remainder = item_idx % (C * outD * outH * outW);
        const int c = remainder / (outD * outH * outW);
        remainder %= (outD * outH * outW);
        const int out_d = remainder / (outH * outW);
        remainder %= (outH * outW);
        const int out_h = remainder / outW;
        const int out_w = remainder % outW;

        // Calculate input window start position
        const int d_start = out_d * 4;
        const int h_start = out_h * 4;
        const int w_start = out_w * 4;

        float val = -FLT_MAX;
        for (int local_d = 0; local_d < 4; ++local_d) {
            for (int local_h = 0; local_h < 4; ++local_h) {
                for (int local_w = 0; local_w < 4; ++local_w) {
                    const int d = d_start + local_d;
                    const int h = h_start + local_h;
                    const int w = w_start + local_w;

                    if (d < D && h < H && w < W) {
                        const int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
                        val = std::max(val, __ldg(&softmax_output[input_idx]));
                    }
                }
            }
        }

        // Warp-level reduction without divergent branches
        for (int offset = 16; offset > 0; offset /= 2) {
            val = std::max(val, __shfl_down_sync(0xffffffff, val, offset));
        }

        // Inter-warp reduction using shared memory
        __shared__ float warp_results[2];
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;

        if (lane_id == 0) {
            warp_results[warp_id] = val;
        }
        __syncthreads();

        // Final reduction in first warp only
        if (warp_id == 0) {
            val = (lane_id < 2) ? warp_results[lane_id] : -FLT_MAX;
            for (int offset = 1; offset > 0; offset >>= 1) {
                val = std::max(val, __shfl_down_sync(0xffffffff, val, offset));
            }

            if (lane_id == 0) {
                const int out_idx = n * (C * outD * outH * outW) + 
                                    c * (outD * outH * outW) + 
                                    out_d * (outH * outW) + 
                                    out_h * outW + 
                                    out_w;
                output[out_idx] = val;
            }
        }
        __syncthreads();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided CPU forward function");
}
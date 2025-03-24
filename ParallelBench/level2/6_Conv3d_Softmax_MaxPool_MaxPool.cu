#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void strided_maxpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int D, const int H, const int W,
    const int outD, const int outH, const int outW,
    const int items_per_block
) {
    // Pre-compute strides for more efficient indexing
    const int stride_n = C * D * H * W;
    const int stride_c = D * H * W;
    const int stride_d = H * W;
    const int stride_h = W;
    
    const int out_stride_c = outD * outH * outW;
    const int out_stride_d = outH * outW;
    
    // Calculate base output position for this block
    const int block_idx = blockIdx.x;
    const int total_elements = N * C * outD * outH * outW;
    const int block_start = block_idx * items_per_block;
    
    // Thread mapping within 4x4x4 window
    const int tid = threadIdx.x;
    const int local_d = tid / 16;
    const int local_h = (tid % 16) / 4;
    const int local_w = tid % 4;

    // Process multiple output elements per block
    #pragma unroll 4
    for (int item_idx = 0; item_idx < items_per_block; item_idx++) {
        const int global_idx = block_start + item_idx;
        if (global_idx >= total_elements) break;

        // Convert linear index to 5D coordinates
        const int n = global_idx / (C * out_stride_c);
        int remainder = global_idx % (C * out_stride_c);
        const int c = remainder / out_stride_c;
        remainder %= out_stride_c;
        const int out_d = remainder / out_stride_d;
        remainder %= out_stride_d;
        const int out_h = remainder / outW;
        const int out_w = remainder % outW;

        // Calculate input window start position
        const int d_start = out_d * 4;
        const int h_start = out_h * 4;
        const int w_start = out_w * 4;

        // Calculate global input position for this thread
        const int d = d_start + local_d;
        const int h = h_start + local_h;
        const int w = w_start + local_w;

        // Load input value using predication
        float val = -FLT_MAX;
        const bool valid = (d < D) && (h < H) && (w < W);
        if (valid) {
            const int input_idx = n * stride_n + c * stride_c + d * stride_d + h * stride_h + w;
            val = __ldg(&input[input_idx]);
        }

        // Warp-level reduction without divergent branches
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            const float other = __shfl_down_sync(0xffffffff, val, offset);
            val = fmaxf(val, other);
        }

        // Inter-warp reduction using shared memory
        __shared__ float warp_results[2];
        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;

        if (lane_id == 0) {
            warp_results[warp_id] = val;
        }
        __syncthreads();

        // Final reduction in first warp only
        if (warp_id == 0) {
            val = (lane_id < 2) ? warp_results[lane_id] : -FLT_MAX;
            
            #pragma unroll
            for (int offset = 1; offset > 0; offset >>= 1) {
                const float other = __shfl_down_sync(0xffffffff, val, offset);
                val = fmaxf(val, other);
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
}

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
    const int threads = 64;

    strided_maxpool_kernel<<<num_blocks, threads>>>(
        softmax_output.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        outD, outH, outW,
        items_per_block
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided CUDA forward function");
}
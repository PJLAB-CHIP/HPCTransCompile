#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

// Host function launching the manually unrolled kernel
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    int pool_kernel_size  // Expected to be 2 for this unrolled kernel
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2); // should be 3

    const int out_h = height - kernel_size + 1;
    const int out_w = width - kernel_size + 1;
    const int pooled_h = (out_h + pool_kernel_size - 1) / pool_kernel_size;
    const int pooled_w = (out_w + pool_kernel_size - 1) / pool_kernel_size;

    auto output = torch::empty({batch_size, out_channels, pooled_h, pooled_w},
                               torch::TensorOptions().device(input.device()).dtype(input.dtype()));

    // Configure 2D block dimensions and 3D grid dimensions
    const int block_dim_x = 16;
    const int block_dim_y = 16;
    const int grid_dim_x = (pooled_w + block_dim_x - 1) / block_dim_x;
    const int grid_dim_y = (pooled_h + block_dim_y - 1) / block_dim_y;
    const int grid_dim_z = batch_size * out_channels;

    #pragma omp parallel for
    for (int bc = 0; bc < batch_size * out_channels; ++bc) {
        int batch = bc / out_channels;
        int channel = bc % out_channels;
        int x = (bc / out_channels) % pooled_w;
        int y = (bc / out_channels) / pooled_w;

        if (y >= pooled_h || x >= pooled_w || batch >= batch_size) continue;

        // For this manually unrolled kernel, we assume pooling kernel size is 2
        int h_start = y * 2;
        int w_start = x * 2;
        float max_val = -1e10f;

        // Manually unrolled 2x2 pooling with four fixed offsets
        // Pool offset (0,0)
        int cur_h = h_start;
        int cur_w = w_start;
        if (cur_h < out_h && cur_w < out_w) {
            float conv = bias[channel];
            // Manually unroll the 3x3 convolution for each in_channel
            for (int ic = 0; ic < in_channels; ic++) {
                int weight_base = ((channel * in_channels) + ic) * 9; // 3x3 = 9
                int input_base = ((batch * in_channels) + ic) * height * width;

                conv += input[input_base + (cur_h + 0) * width + (cur_w + 0)] * weight[weight_base + 0];
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 1)] * weight[weight_base + 1];
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 2)] * weight[weight_base + 2];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 0)] * weight[weight_base + 3];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 1)] * weight[weight_base + 4];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 2)] * weight[weight_base + 5];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 0)] * weight[weight_base + 6];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 1)] * weight[weight_base + 7];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 2)] * weight[weight_base + 8];
            }
            conv -= subtract_value;
            float tmp = conv + 3.0f;
            float clamp_val = std::min(6.0f, std::max(0.0f, tmp));
            float hardswish = conv * clamp_val / 6.0f;
            max_val = std::max(max_val, hardswish);
        }

        // Pool offset (0,1)
        cur_h = h_start;
        cur_w = w_start + 1;
        if (cur_h < out_h && cur_w < out_w) {
            float conv = bias[channel];
            for (int ic = 0; ic < in_channels; ic++) {
                int weight_base = ((channel * in_channels) + ic) * 9;
                int input_base = ((batch * in_channels) + ic) * height * width;
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 0)] * weight[weight_base + 0];
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 1)] * weight[weight_base + 1];
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 2)] * weight[weight_base + 2];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 0)] * weight[weight_base + 3];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 1)] * weight[weight_base + 4];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 2)] * weight[weight_base + 5];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 0)] * weight[weight_base + 6];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 1)] * weight[weight_base + 7];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 2)] * weight[weight_base + 8];
            }
            conv -= subtract_value;
            float tmp = conv + 3.0f;
            float clamp_val = std::min(6.0f, std::max(0.0f, tmp));
            float hardswish = conv * clamp_val / 6.0f;
            max_val = std::max(max_val, hardswish);
        }

        // Pool offset (1,0)
        cur_h = h_start + 1;
        cur_w = w_start;
        if (cur_h < out_h && cur_w < out_w) {
            float conv = bias[channel];
            for (int ic = 0; ic < in_channels; ic++) {
                int weight_base = ((channel * in_channels) + ic) * 9;
                int input_base = ((batch * in_channels) + ic) * height * width;
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 0)] * weight[weight_base + 0];
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 1)] * weight[weight_base + 1];
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 2)] * weight[weight_base + 2];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 0)] * weight[weight_base + 3];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 1)] * weight[weight_base + 4];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 2)] * weight[weight_base + 5];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 0)] * weight[weight_base + 6];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 1)] * weight[weight_base + 7];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 2)] * weight[weight_base + 8];
            }
            conv -= subtract_value;
            float tmp = conv + 3.0f;
            float clamp_val = std::min(6.0f, std::max(0.0f, tmp));
            float hardswish = conv * clamp_val / 6.0f;
            max_val = std::max(max_val, hardswish);
        }

        // Pool offset (1,1)
        cur_h = h_start + 1;
        cur_w = w_start + 1;
        if (cur_h < out_h && cur_w < out_w) {
            float conv = bias[channel];
            for (int ic = 0; ic < in_channels; ic++) {
                int weight_base = ((channel * in_channels) + ic) * 9;
                int input_base = ((batch * in_channels) + ic) * height * width;
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 0)] * weight[weight_base + 0];
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 1)] * weight[weight_base + 1];
                conv += input[input_base + (cur_h + 0) * width + (cur_w + 2)] * weight[weight_base + 2];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 0)] * weight[weight_base + 3];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 1)] * weight[weight_base + 4];
                conv += input[input_base + (cur_h + 1) * width + (cur_w + 2)] * weight[weight_base + 5];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 0)] * weight[weight_base + 6];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 1)] * weight[weight_base + 7];
                conv += input[input_base + (cur_h + 2) * width + (cur_w + 2)] * weight[weight_base + 8];
            }
            conv -= subtract_value;
            float tmp = conv + 3.0f;
            float clamp_val = std::min(6.0f, std::max(0.0f, tmp));
            float hardswish = conv * clamp_val / 6.0f;
            max_val = std.max(max_val, hardswish);
        }

        // Apply Mish activation: x * tanh(softplus(x))
        float softplus = std::log(1.0f + std::exp(max_val));
        float mish = max_val * std::tanh(softplus);

        int out_idx = ((batch * out_channels + channel) * pooled_h + y) * pooled_w + x;
        output[out_idx] = mish;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fully manually unrolled conv-pool-activate forward (CPU)");
}
#include <torch/extension.h>
#include <cmath>
#include <omp.h>

void manually_unrolled_cpu(
    const float* input,    // [batch_size, in_channels, height, width]
    const float* weight,   // [out_channels, in_channels, 3, 3]
    const float* bias,     // [out_channels]
    float* output,         // [batch_size, out_channels, pooled_h, pooled_w]
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const float subtract_val,
    const int out_h,   // out_h = height - 3 + 1
    const int out_w,   // out_w = width  - 3 + 1
    const int pooled_h,
    const int pooled_w
) {
    #pragma omp parallel for collapse(3)
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int channel = 0; channel < out_channels; ++channel) {
            for (int y = 0; y < pooled_h; ++y) {
                for (int x = 0; x < pooled_w; ++x) {
                    int h_start = y * 2;
                    int w_start = x * 2;
                    float max_val = -1e10f;

                    // Pool offset (0,0)
                    int cur_h = h_start;
                    int cur_w = w_start;
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
                        conv -= subtract_val;
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
                        conv -= subtract_val;
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
                        conv -= subtract_val;
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
                        conv -= subtract_val;
                        float tmp = conv + 3.0f;
                        float clamp_val = std::min(6.0f, std::max(0.0f, tmp));
                        float hardswish = conv * clamp_val / 6.0f;
                        max_val = std::max(max_val, hardswish);
                    }

                    // Apply Mish activation: x * tanh(softplus(x))
                    float softplus = logf(1.0f + expf(max_val));
                    float mish = max_val * tanhf(softplus);

                    int out_idx = ((batch * out_channels + channel) * pooled_h + y) * pooled_w + x;
                    output[out_idx] = mish;
                }
            }
        }
    }
}

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

    manually_unrolled_cpu(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        subtract_value,
        out_h,
        out_w,
        pooled_h,
        pooled_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fully manually unrolled conv-pool-activate forward (CPU)");
}
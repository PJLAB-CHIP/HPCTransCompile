#include <torch/extension.h>
#include <omp.h>
#include <cmath>

#define UNROLL_BLOCK_SIZE 256

template<int KERNEL_H, int KERNEL_W>
void unrolled_fused_conv_instnorm_cpu(
    const float* input,
    const float* conv_weight,
    const float* conv_bias,
    float* output,
    const float* inst_scale,
    const float* inst_shift,
    float divide_by,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int output_height,
    int output_width,
    float epsilon) {

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            int out_base = ((n * out_channels + oc) * output_height) * output_width;
            float local_sum = 0.0f;
            float local_sum_sq = 0.0f;

            #pragma omp parallel for reduction(+:local_sum, local_sum_sq)
            for (int idx = 0; idx < output_height * output_width; ++idx) {
                int w_out = idx % output_width;
                int h_out = idx / output_width;
                float conv_val = conv_bias[oc];

                #pragma unroll 4
                for (int ic = 0; ic < in_channels; ++ic) {
                    #pragma unroll
                    for (int i = 0; i < KERNEL_H; ++i) {
                        #pragma unroll
                        for (int j = 0; j < KERNEL_W; ++j) {
                            int in_h = h_out + i;
                            int in_w = w_out + j;
                            int input_idx = ((n * in_channels + ic) * input_height + in_h) * input_width + in_w;
                            int weight_idx = ((oc * in_channels + ic) * KERNEL_H + i) * KERNEL_W + j;
                            conv_val += input[input_idx] * conv_weight[weight_idx];
                        }
                    }
                }

                output[out_base + idx] = conv_val;
                local_sum += conv_val;
                local_sum_sq += conv_val * conv_val;
            }

            float mean = local_sum / (output_height * output_width);
            float variance = local_sum_sq / (output_height * output_width) - mean * mean;
            float inv_std = std::sqrt(variance + epsilon);
            float scale = inst_scale[oc];
            float shift = inst_shift[oc];

            #pragma omp parallel for
            for (int idx = 0; idx < output_height * output_width; ++idx) {
                int out_idx = out_base + idx;
                float val = output[out_idx];
                float norm = (val - mean) * inv_std;
                output[out_idx] = (scale * norm + shift) / divide_by;
            }
        }
    }
}

torch::Tensor forward_cpu(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    c10::optional<torch::Tensor> inst_scale_opt,
    c10::optional<torch::Tensor> inst_shift_opt,
    float divide_by) {

    input = input.contiguous();
    conv_weight = conv_weight.contiguous();
    conv_bias = conv_bias.contiguous();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = conv_weight.size(0);
    int kernel_h = conv_weight.size(2);
    int kernel_w = conv_weight.size(3);

    int output_height = input_height - kernel_h + 1;
    int output_width = input_width - kernel_w + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    torch::Tensor inst_scale, inst_shift;
    if (inst_scale_opt.has_value() && inst_scale_opt.value().defined()) {
        inst_scale = inst_scale_opt.value().contiguous();
    } else {
        inst_scale = torch::ones({out_channels}, output.options());
    }
    if (inst_shift_opt.has_value() && inst_shift_opt.value().defined()) {
        inst_shift = inst_shift_opt.value().contiguous();
    } else {
        inst_shift = torch::zeros({out_channels}, output.options());
    }

    if (kernel_h == 3 && kernel_w == 3) {
        unrolled_fused_conv_instnorm_cpu<3, 3>(
            input.data_ptr<float>(),
            conv_weight.data_ptr<float>(),
            conv_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            inst_scale.data_ptr<float>(),
            inst_shift.data_ptr<float>(),
            divide_by,
            batch_size,
            in_channels,
            input_height,
            input_width,
            out_channels,
            output_height,
            output_width,
            1e-5f);
    } else if (kernel_h == 5 && kernel_w == 5) {
        unrolled_fused_conv_instnorm_cpu<5, 5>(
            input.data_ptr<float>(),
            conv_weight.data_ptr<float>(),
            conv_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            inst_scale.data_ptr<float>(),
            inst_shift.data_ptr<float>(),
            divide_by,
            batch_size,
            in_channels,
            input_height,
            input_width,
            out_channels,
            output_height,
            output_width,
            1e-5f);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Unrolled Fused Conv2d + InstanceNorm + Division (CPU)");
}
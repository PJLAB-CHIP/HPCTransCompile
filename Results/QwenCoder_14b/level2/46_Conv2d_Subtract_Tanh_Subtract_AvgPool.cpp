#include <torch/extension.h>
#include <omp.h>
#include <cmath>

// CPU function performing the same operation as the CUDA kernel

void process_cpu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int pool_size,
    const int pool_out_h,
    const int pool_out_w,
    const float subtract1,
    const float subtract2
) {
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < out_channels; c++) {
            for (int ph = 0; ph < pool_out_h; ph++) {
                for (int pw = 0; pw < pool_out_w; pw++) {
                    const int h_start = ph * pool_size;
                    const int w_start = pw * pool_size;

                    float pool_sum = 0.0f;
                    int pool_count = 0;

                    for (int ph_offset = 0; ph_offset < pool_size; ph_offset++) {
                        for (int pw_offset = 0; pw_offset < pool_size; pw_offset++) {
                            const int h = h_start + ph_offset;
                            const int w = w_start + pw_offset;
                            if (h >= out_height || w >= out_width) continue;

                            float conv_result = bias[c];

                            for (int ic = 0; ic < in_channels; ic++) {
                                for (int kh = 0; kh < kernel_size; kh++) {
                                    for (int kw = 0; kw < kernel_size; kw++) {
                                        const int in_h = h + kh;
                                        const int in_w = w + kw;
                                        if (in_h < in_height && in_w < in_width) {
                                            const int in_idx = b * (in_channels * in_height * in_width) +
                                                               ic * (in_height * in_width) +
                                                               in_h * in_width + in_w;
                                            const int w_idx = c * (in_channels * kernel_size * kernel_size) +
                                                              ic * (kernel_size * kernel_size) +
                                                              kh * kernel_size + kw;
                                            conv_result += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }

                            conv_result = tanhf(conv_result - subtract1);
                            conv_result = conv_result - subtract2;

                            pool_sum += conv_result;
                            pool_count++;
                        }
                    }

                    if (pool_count > 0) {
                        const int idx = b * (out_channels * pool_out_h * pool_out_w) +
                                       c * (pool_out_h * pool_out_w) +
                                       ph * pool_out_w + pw;
                        output[idx] = pool_sum / pool_count;
                    }
                }
            }
        }
    }
}

// Host function invoking the CPU function

torch::Tensor forward(
    torch::Tensor input,
    int kernel_size_pool,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    float subtract1_value,
    float subtract2_value
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = conv_weight.size(0);
    const int kernel_size = conv_weight.size(2);
    
    const int out_height = in_height - kernel_size + 1;
    const int out_width = in_width - kernel_size + 1;
    const int pool_out_h = out_height / kernel_size_pool;
    const int pool_out_w = out_width / kernel_size_pool;

    auto output = torch::zeros({batch_size, out_channels, pool_out_h, pool_out_w}, input.options());

    process_cpu(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        kernel_size_pool,
        pool_out_h,
        pool_out_w,
        subtract1_value,
        subtract2_value
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced Conv+sub+tanh+sub+pool forward");
}
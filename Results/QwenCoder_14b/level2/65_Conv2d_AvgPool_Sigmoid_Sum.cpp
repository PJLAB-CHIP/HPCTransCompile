#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define POOL_SIZE 2

void conv_pool_sigmoid_sum_cpu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size
) {
    const int out_h = height - kernel_size + 1;
    const int out_w = width - kernel_size + 1;
    const int pool_h = out_h / POOL_SIZE;
    const int pool_w = out_w / POOL_SIZE;
    const int total_work = out_channels * pool_h * pool_w;
    const float pool_scale = 1.0f / (POOL_SIZE * POOL_SIZE);

    #pragma omp parallel for
    for (int bid = 0; bid < batch_size; ++bid) {
        float thread_sum = 0.0f;

        for (int idx = 0; idx < total_work; ++idx) {
            const int oc = idx / (pool_h * pool_w);
            const int ph = idx % (pool_h * pool_w);
            const int pool_row = (ph / pool_w) * POOL_SIZE;
            const int pool_col = (ph % pool_w) * POOL_SIZE;

            float conv_val = bias[oc];

            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    const int h_in = pool_row + kh;
                    const float* input_row = &input[((bid * in_channels + ic) * height + h_in) * width];
                    const float* weight_row = &weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size];

                    for (int kw = 0; kw < kernel_size; ++kw) {
                        conv_val += input_row[pool_col + kw] * weight_row[kw];
                    }
                }
            }

            conv_val *= pool_scale;
            thread_sum += 1.0f / (1.0f + expf(-conv_val));
        }

        output[bid] = thread_sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    auto output = torch::empty({batch_size}, input.options());

    conv_pool_sigmoid_sum_cpu(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CPU Conv+Pool+Sigmoid+Sum");
}
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Host function invoking the CPU implementation

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

    const int total_elements = batch_size * out_channels * pool_out_h * pool_out_w;

    #pragma omp parallel for
    for (int idx = 0; idx < total_elements; ++idx) {
        const int pw = idx % pool_out_w;
        const int ph = (idx / pool_out_w) % pool_out_h;
        const int c = (idx / (pool_out_w * pool_out_h)) % out_channels;
        const int b = idx / (pool_out_w * pool_out_h * out_channels);

        const int h_start = ph * kernel_size_pool;
        const int w_start = pw * kernel_size_pool;

        float pool_sum = 0.0f;
        int pool_count = 0;

        for (int ph_offset = 0; ph_offset < kernel_size_pool; ph_offset++) {
            for (int pw_offset = 0; pw_offset < kernel_size_pool; pw_offset++) {
                const int h = h_start + ph_offset;
                const int w = w_start + pw_offset;
                if (h >= out_height || w >= out_width) continue;

                float conv_result = conv_bias[c];

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
                                conv_result += input[in_idx].item<float>() * conv_weight[c * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + kh * kernel_size + kw].item<float>();
                            }
                        }
                    }
                }

                conv_result = tanhf(conv_result - subtract1_value);
                conv_result = conv_result - subtract2_value;

                pool_sum += conv_result;
                pool_count++;
            }
        }

        if (pool_count > 0) {
            output[idx] = pool_sum / pool_count;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced Conv+sub+tanh+sub+pool forward");
}
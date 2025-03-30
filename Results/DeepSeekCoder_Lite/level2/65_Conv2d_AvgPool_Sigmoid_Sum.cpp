#include <torch/extension.h>
#include <vector>
#include <omp.h>

#define BLOCK_SIZE 512
#define POOL_SIZE 2

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

    #pragma omp parallel for
    for (int bid = 0; bid < batch_size; ++bid) {
        const int out_h = height - kernel_size + 1;
        const int out_w = width - kernel_size + 1;
        const int pool_h = out_h / POOL_SIZE;
        const int pool_w = out_w / POOL_SIZE;
        const int total_work = out_channels * pool_h * pool_w;

        const float pool_scale = 1.0f / (POOL_SIZE * POOL_SIZE);
        float thread_sum = 0.0f;

        for (int idx = omp_get_thread_num() * (total_work / omp_get_num_threads()); idx < total_work; idx += total_work / omp_get_num_threads()) {
            const int oc = idx / (pool_h * pool_w);
            const int ph = idx % (pool_h * pool_w);
            const int pool_row = (ph / pool_w) * POOL_SIZE;
            const int pool_col = (ph % pool_w) * POOL_SIZE;
            
            float conv_val = bias[oc];

            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < 3; ++kh) {
                    const int h_in = pool_row + kh;
                    const float* input_row = &input[((bid * in_channels + ic) * height + h_in) * width];
                    const float* weight_row = &weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size];
                    
                    for (int kw = 0; kw < 3; ++kw) {
                        conv_val = conv_val + input_row[pool_col + kw] * weight_row[kw];
                    }
                }
            }

            conv_val *= pool_scale;        
            thread_sum += 1.0f / (1.0f + exp(-conv_val));
        }

        output[bid] = thread_sum;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "BLOCK512 Conv+Pool+Sigmoid+Sum");
}
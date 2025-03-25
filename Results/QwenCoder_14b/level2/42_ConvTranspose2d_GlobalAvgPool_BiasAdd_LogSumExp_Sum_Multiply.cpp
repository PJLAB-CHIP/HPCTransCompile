#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <cmath>

const int BLOCK_SIZE = 256;

void sync_optimized_cpu_kernel(
    const float* input,
    const float* conv_weights,
    const float* conv_bias,
    const float* bias,
    float* output,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    #pragma omp parallel for collapse(4) num_threads(omp_get_num_procs())
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = b * channels * height * width + c * height * width + h * width + w;
                    float val = input[idx];
                    for (int i = 0; i < channels; ++i) {
                        val += conv_weights[i] * val;
                    }
                    val += conv_bias[c];
                    
                    // Accumulate results in shared memory
                    #pragma omp atomic
                    output[b] += val / (height * width);
                }
            }
        }
    }

    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (int c = 0; c < channels; ++c) {
        output[c] += bias[c];
        output[c] = expf(output[c]);
    }

    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sum += output[c];
        }
        output[b] = logf(sum) * 10.0f;
    }
}

torch::Tensor module_fn(
    torch::Tensor x,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor");
    TORCH_CHECK(!conv_transpose.is_cuda(), "conv_transpose must be a CPU tensor");
    TORCH_CHECK(!conv_transpose_bias.is_cuda(), "conv_transpose_bias must be a CPU tensor");
    TORCH_CHECK(!bias.is_cuda(), "bias must be a CPU tensor");
    
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    
    auto output = torch::empty({batch_size, 1}, x.options());
    
    sync_optimized_cpu_kernel(
        x.data_ptr<float>(),
        conv_transpose.data_ptr<float>(),
        conv_transpose_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Sync optimized CPU kernel forward");
}
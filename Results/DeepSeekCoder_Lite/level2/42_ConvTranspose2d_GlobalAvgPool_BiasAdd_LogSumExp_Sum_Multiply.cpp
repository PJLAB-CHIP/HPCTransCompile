#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>
#include <cmath>

const int BLOCK_SIZE = 256;

void sync_optimized_kernel(
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
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < channels; ++c) {
                    int idx = b * channels * height * width + c * height * width + h * width + w;
                    float val = input[idx];
                    for (int i = 0; i < channels; ++i) {
                        val += conv_weights[i] * val;
                    }
                    val += conv_bias[idx % channels];
                    
                    // Utilize shared memory for intermediate results without unnecessary synchronizations
                    #pragma omp atomic
                    output[b] += exp(val / (height * width) + bias[c]);
                }
            }
        }
    }
}

torch::Tensor module_fn(
    torch::Tensor x,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(conv_transpose.is_cpu(), "conv_transpose must be a CPU tensor");
    TORCH_CHECK(conv_transpose_bias.is_cpu(), "conv_transpose_bias must be a CPU tensor");
    TORCH_CHECK(bias.is_cpu(), "bias must be a CPU tensor");
    
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    
    auto output = torch::empty({batch_size, 1}, x.options());
    
    sync_optimized_kernel(
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
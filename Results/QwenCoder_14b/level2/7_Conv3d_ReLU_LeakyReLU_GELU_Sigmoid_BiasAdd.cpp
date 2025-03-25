#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define WARP_SIZE 32
#define BLOCK_SIZE 256

float process_value(float val, const float* bias, int bias_idx) {
    // ReLU
    val = std::max(0.0f, val);
    
    // LeakyReLU
    val = std::max(0.01f * val, val);
    
    // GELU
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    val = 0.5f * val * (1.0f + std::tanh(sqrt_2_over_pi * (val + 0.044715f * std::pow(val, 3.0f))));
    
    // Sigmoid
    val = 1.0f / (1.0f + std::exp(-val));
    
    // Add bias
    val += bias[bias_idx];
    
    return val;
}

void apply_activations_and_bias_cpu(
    float* output, const float* bias,
    int batch_size, int out_channels, int depth, int height, int width
) {
    const int spatial_size = depth * height * width;
    const int elements_per_channel = spatial_size;
    const int total_elements = batch_size * out_channels * spatial_size;

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < out_channels; ++c) {
            for (int d = 0; d < depth; ++d) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = b * out_channels * spatial_size + c * spatial_size + d * height * width + h * width + w;
                        output[idx] = process_value(output[idx], bias, c);
                    }
                }
            }
        }
    }
}

torch::Tensor module_fn_cpu(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias
) {
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(conv_weight);
    CHECK_CONTIGUOUS(conv_bias);
    CHECK_CONTIGUOUS(bias);

    auto output = torch::conv3d(x, conv_weight, conv_bias);

    int batch_size = output.size(0);
    int out_channels = output.size(1);
    int depth = output.size(2);
    int height = output.size(3);
    int width = output.size(4);

    apply_activations_and_bias_cpu(
        output.data_ptr<float>(), bias.data_ptr<float>(),
        batch_size, out_channels, depth, height, width
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu, "CPU implementation of module_fn");
}
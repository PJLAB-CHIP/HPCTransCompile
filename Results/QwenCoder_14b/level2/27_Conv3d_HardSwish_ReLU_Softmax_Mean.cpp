#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Constants for HardSwish
const float hswish_offset = 3.0f;
const float hswish_cap = 6.0f;
const float hswish_div = 1.0f / 6.0f;

// Function to apply HardSwish, ReLU, and Softmax
void fused_cpu_kernel(const float* input, float* output, int batch_size, int channels, int spatial_size) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < spatial_size; ++s) {
            float max_val = -std::numeric_limits<float>::max();

            // Pass 1: Compute maximum activation value across channels for numerical stability
            for (int c = 0; c < channels; ++c) {
                int idx = (b * channels + c) * spatial_size + s;
                float x = input[idx];
                float relu6 = std::fmin(std::fmax(x + hswish_offset, 0.0f), hswish_cap);
                float hswish = x * relu6 * hswish_div;
                float act = std::fmax(hswish, 0.0f);
                if (act > max_val) {
                    max_val = act;
                }
            }

            float sum_exp = 0.0f;

            // Pass 2: Compute exponentials and accumulate the sum, store exp values temporarily in output
            for (int c = 0; c < channels; ++c) {
                int idx = (b * channels + c) * spatial_size + s;
                float x = input[idx];
                float relu6 = std::fmin(std::fmax(x + hswish_offset, 0.0f), hswish_cap);
                float hswish = x * relu6 * hswish_div;
                float act = std::fmax(hswish, 0.0f);
                float exp_val = std::exp(act - max_val);
                sum_exp += exp_val;
                output[idx] = exp_val;
            }

            // Pass 3: Normalize the exponentials to obtain softmax probabilities
            for (int c = 0; c < channels; ++c) {
                int idx = (b * channels + c) * spatial_size + s;
                output[idx] = output[idx] / sum_exp;
            }
        }
    }
}

// Module forward function: combines conv3d, the fused activation and softmax kernel, and mean reduction
torch::Tensor module_forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias) {
    // Ensure tensors are contiguous and on CPU
    x = x.contiguous().cpu();
    conv_weight = conv_weight.contiguous().cpu();
    conv_bias = conv_bias.contiguous().cpu();

    // Perform 3D convolution via PyTorch's conv3d
    x = torch::conv3d(x, conv_weight, conv_bias);

    // Retrieve tensor dimensions
    int64_t batch_size = x.size(0);
    int64_t channels = x.size(1);
    int64_t depth = x.size(2);
    int64_t height = x.size(3);
    int64_t width = x.size(4);
    int64_t spatial_size = depth * height * width;

    // Allocate intermediate tensor for softmax result
    torch::Tensor x_softmax = torch::empty_like(x);

    // Apply fused kernel
    fused_cpu_kernel(x.data_ptr<float>(), x_softmax.data_ptr<float>(), batch_size, channels, spatial_size);

    // Reshape back to original dimensions and compute mean over spatial dims
    torch::Tensor output = x_softmax.view({batch_size, channels, depth, height, width}).mean({2, 3, 4});
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_forward, "Fused CPU module forward with OpenMP parallelization");
}
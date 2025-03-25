#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <omp.h>

// Define the CPU version of the post-processing function
void cpu_post_process(
    float* output,
    const int total_size,
    const int height,
    const int width,
    const int channels,
    const float scaling_factor,
    const float* global_bias
) {
    // Copy bias values to a local array for faster access
    std::vector<float> s_bias(channels);
    for (int i = 0; i < channels; ++i) {
        s_bias[i] = global_bias[i];
    }

    // Compute the number of vectorized (float4) groups and the remaining elements
    int vec_size = total_size / 4;
    int remainder = total_size % 4;

    int hw_size = height * width;

    #pragma omp parallel for
    for (int i = 0; i < vec_size; ++i) {
        int base_index = i * 4;
        float results[4];

        // Process each element of the float4
        {
            int index = base_index;
            int c = (index / hw_size) % channels;
            float val = output[index] + s_bias[c];
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            val = val * scaling_factor;
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            results[0] = val / scaling_factor;
        }
        {
            int index = base_index + 1;
            int c = (index / hw_size) % channels;
            float val = output[index] + s_bias[c];
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            val = val * scaling_factor;
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            results[1] = val / scaling_factor;
        }
        {
            int index = base_index + 2;
            int c = (index / hw_size) % channels;
            float val = output[index] + s_bias[c];
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            val = val * scaling_factor;
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            results[2] = val / scaling_factor;
        }
        {
            int index = base_index + 3;
            int c = (index / hw_size) % channels;
            float val = output[index] + s_bias[c];
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            val = val * scaling_factor;
            val = fminf(fmaxf(val, 0.0f), 1.0f);
            results[3] = val / scaling_factor;
        }

        // Write the processed values back
        output[base_index] = results[0];
        output[base_index + 1] = results[1];
        output[base_index + 2] = results[2];
        output[base_index + 3] = results[3];
    }

    // Process any remaining elements that weren't covered by the vectorized loop
    #pragma omp parallel for
    for (int i = 0; i < remainder; ++i) {
        int index = vec_size * 4 + i;
        int c = (index / hw_size) % channels;
        float val = output[index] + s_bias[c];
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val * scaling_factor;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        output[index] = val / scaling_factor;
    }
}

// Forward function performs conv_transpose2d followed by the post-processing kernel

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    float scaling_factor,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    // Perform transposed convolution using PyTorch's built-in function
    auto output = torch::conv_transpose2d(
        x, conv_transpose, conv_transpose_bias,
        stride, padding, output_padding
    );

    const int batch_size = output.size(0);
    const int channels = output.size(1);
    const int height = output.size(2);
    const int width = output.size(3);
    const int total_size = batch_size * channels * height * width;

    // Call the CPU post-processing function
    cpu_post_process(
        output.data_ptr<float>(),
        total_size,
        height,
        width,
        channels,
        scaling_factor,
        bias.data_ptr<float>()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CPU version of the shared-memory bias optimized post-processing kernel");
}
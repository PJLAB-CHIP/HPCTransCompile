#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <omp.h>

// Function to compute the 5D tensor indices given the flattened index idx
void compute_indices(int64_t idx, int64_t width, int64_t height, int64_t depth, int64_t channels, int64_t& w, int64_t& h, int64_t& d, int64_t& c) {
    w = idx % width;
    h = (idx / width) % height;
    d = (idx / (width * height)) % depth;
    c = (idx / (width * height * depth)) % channels;
}

// Function to apply LeakyReLU and GELU activation
float apply_activation(float x, const float* sum_tensor, int64_t c) {
    float y = fmaxf(x, 0.2f * x);
    y += sum_tensor[c];
    y = fmaxf(fminf(y, 1.0f), -1.0f);
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (y + 0.044715f * y * y * y)));
    return y * cdf;
}

// Main function to process the tensor in parallel using OpenMP
void my_cpu_function(
    torch::Tensor& x,
    torch::Tensor& sum_tensor) {

    const int64_t num_elements = x.numel();
    const int64_t batch_size = x.size(0);
    const int64_t channels = x.size(1);
    const int64_t depth = x.size(2);
    const int64_t height = x.size(3);
    const int64_t width = x.size(4);

    // Use OpenMP for parallelization
    #pragma omp parallel for
    for (int64_t idx = 0; idx < num_elements; ++idx) {
        int64_t w, h, d, c;
        compute_indices(idx, width, height, depth, channels, w, h, d, c);
        float x_val = x[idx];
        float result = apply_activation(x_val, sum_tensor.data_ptr<float>(), c);
        x[idx] = result;
    }
}

// Forward function that applies the custom CPU function
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor sum_tensor) {

    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor");
    TORCH_CHECK(!conv_weight.is_cuda(), "conv_weight must be a CPU tensor");
    TORCH_CHECK(!conv_bias.is_cuda(), "conv_bias must be a CPU tensor");
    TORCH_CHECK(!sum_tensor.is_cuda(), "sum_tensor must be a CPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be of type float32");

    // Perform 3D convolution
    auto x_conv = at::conv3d(x, conv_weight, conv_bias);

    // Ensure output is contiguous
    auto output = x_conv.contiguous();

    // Apply the optimized CPU function
    my_cpu_function(output, sum_tensor);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom forward function (CPU)");
}
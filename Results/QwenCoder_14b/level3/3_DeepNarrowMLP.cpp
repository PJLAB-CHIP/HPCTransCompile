#include <torch/extension.h>
#include <vector>
#include <cmath>

// Define the number of threads per block used for intra-block reduction
constexpr int THREADS_PER_BLOCK = 128;

// Optimized CPU function with ReLU activation
template <typename T>
void optimized_linear_relu_cpu(
    const T* input,
    int in_dim,
    const T* weight,
    const T* bias,
    T* output,
    int batch_size,
    int out_dim) {

    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int j = 0; j < out_dim; ++j) {
            T sum = (T)0;
            for (int t = 0; t < in_dim; ++t) {
                sum += input[batch * in_dim + t] * weight[j * in_dim + t];
            }
            T z = sum + bias[j];
            output[batch * out_dim + j] = z > (T)0 ? z : (T)0; // ReLU activation
        }
    }
}

// Optimized CPU function without activation
template <typename T>
void optimized_linear_cpu(
    const T* input,
    int in_dim,
    const T* weight,
    const T* bias,
    T* output,
    int batch_size,
    int out_dim) {

    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int j = 0; j < out_dim; ++j) {
            T sum = (T)0;
            for (int t = 0; t < in_dim; ++t) {
                sum += input[batch * in_dim + t] * weight[j * in_dim + t];
            }
            output[batch * out_dim + j] = sum + bias[j];
        }
    }
}

// The forward function iterates through layers of the MLP
torch::Tensor forward(
    torch::Tensor x,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases) {

    TORCH_CHECK(weights.size() == biases.size(), "Weights and biases count mismatch");
    TORCH_CHECK(x.size(1) == weights[0].size(1), "Input dimension mismatch");

    torch::Tensor current_input = x;

    // Process all layers except the last one with ReLU activation
    for (size_t i = 0; i < weights.size() - 1; i++) {
        auto weight = weights[i];
        auto bias = biases[i];
        int in_dim = weight.size(1);
        int out_dim = weight.size(0);
        int batch_size = current_input.size(0);

        auto output = torch::zeros({batch_size, out_dim}, 
            torch::device(torch::kCPU).dtype(current_input.dtype()));

        if (current_input.dtype() == torch::kFloat32) {
            optimized_linear_relu_cpu<float>(
                current_input.data_ptr<float>(),
                in_dim,
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                out_dim);
        } else {
            TORCH_CHECK(false, "Unsupported dtype");
        }
        current_input = output;
    }

    // Last layer without ReLU activation
    auto weight = weights.back();
    auto bias = biases.back();
    int in_dim = weight.size(1);
    int out_dim = weight.size(0);
    int batch_size = current_input.size(0);

    auto output = torch::zeros({batch_size, out_dim}, 
        torch::device(torch::kCPU).dtype(current_input.dtype()));

    if (current_input.dtype() == torch::kFloat32) {
        optimized_linear_cpu<float>(
            current_input.data_ptr<float>(),
            in_dim,
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            out_dim);
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized MLP forward (CPU)");
}
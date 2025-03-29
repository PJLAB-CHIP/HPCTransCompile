#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Function to compute the softmax of a single row
void softmax_row(const float* x, float* y, int num_features) {
    // Find the maximum value in the row
    float max_val = -INFINITY;
    for (int i = 0; i < num_features; ++i) {
        max_val = std::max(max_val, x[i]);
    }

    // Compute exponentials and accumulate partial sums
    float sum_val = 0.0f;
    for (int i = 0; i < num_features; ++i) {
        y[i] = std::exp(x[i] - max_val);
        sum_val += y[i];
    }

    // Normalize the results
    for (int i = 0; i < num_features; ++i) {
        y[i] /= sum_val;
    }
}

// Host function to apply softmax to each row in parallel
void softmax_forward_cpu(const float* x, float* y, int batch_size, int num_features) {
    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const float* x_row = x + batch_idx * num_features;
        float* y_row = y + batch_idx * num_features;
        softmax_row(x_row, y_row, num_features);
    }
}

// C++ forward function exposed to PyTorch
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(!x.is_cuda(), "Input tensor must be a CPU tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);

    auto y = torch::empty_like(x);
    softmax_forward_cpu(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CPU)");
}

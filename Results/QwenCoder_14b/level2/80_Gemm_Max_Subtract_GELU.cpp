#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <omp.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_DIM 32  // Aligned with warp size

inline float gelu(float x) {
    const float a = 0.797884560802865f;
    const float b = 0.044715f;
    float cdf = 0.5f * (1.0f + tanhf(a * (x + b * x * x * x)));
    return x * cdf;
}

void warp_aligned_gemm_cpu(const float* x, const float* weight, const float* bias, float* y, int batch, int in_features, int out_features) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < out_features; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                sum += x[row * in_features + k] * weight[col * in_features + k];
            }
            y[row * out_features + col] = sum + bias[col];
        }
    }
}

void warp_reduce_max_cpu(const float* input, float* output, int rows, int cols, int reduce_dim) {
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        float max_val = -std::numeric_limits<float>::max();
        for (int j = 0; j < cols; ++j) {
            max_val = std::max(max_val, input[i * cols + j]);
        }
        output[i] = max_val;
    }
}

void warp_fused_mean_gelu_cpu(float* data, int rows, int cols) {
    #pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += data[row * cols + col];
        }
        float mean = sum / cols;
        for (int col = 0; col < cols; ++col) {
            float val = data[row * cols + col] - mean;
            data[row * cols + col] = gelu(val);
        }
    }
}

torch::Tensor forward(torch::Tensor x, int max_dim, torch::Tensor weight, torch::Tensor bias) {
    const int batch = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    auto y = torch::empty({batch, out_features}, x.options());

    warp_aligned_gemm_cpu(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), batch, in_features, out_features);

    auto max_out = (max_dim == 0) ?
        torch::empty({1, out_features}, y.options()) :
        torch::empty({batch, 1}, y.options());

    const int rows = (max_dim == 0) ? batch : 1;
    const int cols = (max_dim == 0) ? out_features : batch;

    warp_reduce_max_cpu(y.data_ptr<float>(), max_out.data_ptr<float>(), rows, cols, max_dim);

    const int final_rows = max_out.size(0);
    const int final_cols = max_out.size(1);

    warp_fused_mean_gelu_cpu(max_out.data_ptr<float>(), final_rows, final_cols);

    return max_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CPU forward implementation");
}
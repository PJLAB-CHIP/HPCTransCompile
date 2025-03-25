#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <omp.h>

// Fused CPU kernel: Applies ReLU and then performs 2D max pooling in one pass.
void fused_relu_pool_cpu(
    const float* input,
    float* output,
    int batch, int channels,
    int height, int width,
    int pool_h, int pool_w, int stride
) {
    int out_h = (height - pool_h) / stride + 1;
    int out_w = (width - pool_w) / stride + 1;
    int total = batch * channels * out_h * out_w;

    #pragma omp parallel for
    for (int idx = 0; idx < total; idx++) {
        int tmp = idx;
        int w = tmp % out_w; tmp /= out_w;
        int h = tmp % out_h; tmp /= out_h;
        int c = tmp % channels; tmp /= channels;
        int b = tmp;

        int in_row_start = h * stride;
        int in_col_start = w * stride;
        float max_val = 0.0f;

        for (int i = 0; i < pool_h; i++) {
            for (int j = 0; j < pool_w; j++) {
                int in_row = in_row_start + i;
                int in_col = in_col_start + j;
                float val = input[((b * channels + c) * height + in_row) * width + in_col];
                float relu_val = std::max(val, 0.0f);
                if (relu_val > max_val) {
                    max_val = relu_val;
                }
            }
        }
        output[idx] = max_val;
    }
}

// Simple flattening function
void flatten_cpu(const float* input, float* output, int total) {
    #pragma omp parallel for
    for (int idx = 0; idx < total; idx++) {
        output[idx] = input[idx];
    }
}

// Forward function for the LeNet-5 network on CPU
torch::Tensor forward_cpu(
    torch::Tensor x,
    torch::Tensor conv1_weight, torch::Tensor conv1_bias,
    torch::Tensor conv2_weight, torch::Tensor conv2_bias,
    torch::Tensor fc1_weight, torch::Tensor fc1_bias,
    torch::Tensor fc2_weight, torch::Tensor fc2_bias,
    torch::Tensor fc3_weight, torch::Tensor fc3_bias
) {
    // First Convolutional Layer
    auto conv1 = torch::conv2d(x, conv1_weight, conv1_bias, {1, 1});

    // Fused ReLU and Pooling
    int B = conv1.size(0);
    int C = conv1.size(1);
    int H = conv1.size(2);
    int W = conv1.size(3);
    int pool_h = 2, pool_w = 2, stride = 2;
    int out_h = (H - pool_h) / stride + 1;
    int out_w = (W - pool_w) / stride + 1;

    auto pool1 = torch::empty({B, C, out_h, out_w}, conv1.options());
    int total_pool1 = B * C * out_h * out_w;
    fused_relu_pool_cpu(
        conv1.data_ptr<float>(), pool1.data_ptr<float>(), B, C, H, W, pool_h, pool_w, stride);

    // Second Convolutional Layer
    auto conv2 = torch::conv2d(pool1, conv2_weight, conv2_bias, {1, 1});
    B = conv2.size(0);
    C = conv2.size(1);
    H = conv2.size(2);
    W = conv2.size(3);
    out_h = (H - pool_h) / stride + 1;
    out_w = (W - pool_w) / stride + 1;
    auto pool2 = torch::empty({B, C, out_h, out_w}, conv2.options());
    int total_pool2 = B * C * out_h * out_w;
    fused_relu_pool_cpu(
        conv2.data_ptr<float>(), pool2.data_ptr<float>(), B, C, H, W, pool_h, pool_w, stride);

    // Flatten the output
    auto flat = pool2.view({pool2.size(0), -1});

    // Fully connected layers
    auto fc1 = torch::linear(flat, fc1_weight, fc1_bias);
    fc1 = torch::relu(fc1);
    auto fc2 = torch::linear(fc1, fc2_weight, fc2_bias);
    fc2 = torch::relu(fc2);
    auto fc3 = torch::linear(fc2, fc3_weight, fc3_bias);

    return fc3;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "LeNet-5 forward pass with fused ReLU and pooling on CPU");
}
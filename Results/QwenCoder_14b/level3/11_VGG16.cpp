#include <torch/extension.h>
#include <omp.h>

#define TILE_SIZE 16

// CPU version of the 2D convolution with atomic operations optimization
void conv2d_cpu(const float* input, const float* weight, const float* bias,
                 float* output, int N, int C, int H, int W,
                 int K, int P, int stride) {
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float sum = bias[k];
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < 3; ++kh) {
                            for (int kw = 0; kw < 3; ++kw) {
                                int ih = h - P + kh;
                                int iw = w - P + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float in_val = input[n * C * H * W + c * H * W + ih * W + iw];
                                    float weight_val = weight[k * C * 3 * 3 + c * 3 * 3 + kh * 3 + kw];
                                    sum += in_val * weight_val;
                                }
                            }
                        }
                    }
                    output[n * K * H * W + k * H * W + h * W + w] = sum;
                }
            }
        }
    }
}

// Optimized VGG16 forward pass using the custom optimized convolution
torch::Tensor optimized_vgg16_forward_cpu(
    torch::Tensor x,
    std::vector<torch::Tensor> conv_weights,
    std::vector<torch::Tensor> conv_biases,
    std::vector<torch::Tensor> fc_weights,
    std::vector<torch::Tensor> fc_biases,
    bool is_training
) {
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = conv_weights[0].size(0);
    const int P = conv_weights[0].size(2) / 2;

    auto output = torch::empty({N, K, H, W}, x.options());

    conv2d_cpu(x.data_ptr<float>(), conv_weights[0].data_ptr<float>(), conv_biases[0].data_ptr<float>(),
               output.data_ptr<float>(), N, C, H, W, K, P, 1);

    auto current = torch::relu(output);

    for (int i = 1; i < 13; ++i) {
        current = torch::conv2d(current, conv_weights[i], conv_biases[i], /*stride=*/1, /*padding=*/1);
        current = torch::relu(current);
        // Apply max pooling after every block except the first layer of block 1
        if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
            current = torch::max_pool2d(current, /*kernel_size=*/2, /*stride=*/2);
        }
    }

    current = current.flatten(1);
    current = torch::linear(current, fc_weights[0], fc_biases[0]);
    current = torch::relu(current);
    if (is_training) {
        current = torch::dropout(current, /*p=*/0.0, /*train=*/true);
    }
    current = torch::linear(current, fc_weights[1], fc_biases[1]);
    current = torch::relu(current);
    if (is_training) {
        current = torch::dropout(current, /*p=*/0.0, /*train=*/true);
    }
    current = torch::linear(current, fc_weights[2], fc_biases[2]);

    return current;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_vgg16_forward_cpu, "Optimized VGG16 forward (CPU)");
}
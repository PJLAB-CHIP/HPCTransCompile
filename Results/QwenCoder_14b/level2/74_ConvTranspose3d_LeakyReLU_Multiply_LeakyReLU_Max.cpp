#include <torch/extension.h>
#include <omp.h>

void fused_leaky_relu_multiply_cpu(
    float* output,
    const float* input,
    const float* multiplier,
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const float negative_slope)
{
    #pragma omp parallel for collapse(5)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int d = 0; d < D; d++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        const int idx = ((n * C + c) * D + d) * H * W + h * W + w;
                        float val = input[idx];
                        
                        // First LeakyReLU
                        val = val > 0 ? val : val * negative_slope;
                        
                        // Multiplication
                        val *= multiplier[c];
                        
                        // Second LeakyReLU
                        val = val > 0 ? val : val * negative_slope;
                        
                        output[idx] = val;
                    }
                }
            }
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    at::Tensor conv_transpose,
    at::Tensor conv_transpose_bias,
    at::Tensor multiplier)
{
    // Transposed convolution
    auto conv_out = at::conv_transpose3d(
        x,
        conv_transpose,
        conv_transpose_bias,
        /*stride=*/{stride, stride, stride},
        /*padding=*/{padding, padding, padding},
        /*output_padding=*/{output_padding, output_padding, output_padding},
        /*groups=*/1,
        /*dilation=*/1
    );

    auto output = at::empty_like(conv_out);
    
    const int N = conv_out.size(0);
    const int C = conv_out.size(1);
    const int D = conv_out.size(2);
    const int H = conv_out.size(3);
    const int W = conv_out.size(4);

    fused_leaky_relu_multiply_cpu(
        output.data_ptr<float>(),
        conv_out.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        N, C, D, H, W,
        0.2f
    );

    // Max Pooling (kernel_size=2)
    return at::max_pool3d(output, {2, 2, 2});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward pass for module_fn (CPU)");
}
#include <torch/extension.h>
#include <vector>
#include <omp.h>

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

    const float negative_slope = 0.2f;

    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int z = 0; z < D; z++) {
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        const int idx = ((n * C + c) * D + z) * H * W + y * W + x;
                        float val = conv_out.data_ptr<float>()[idx];
                        
                        // First LeakyReLU
                        val = val > 0 ? val : val * negative_slope;
                        
                        // Multiplication
                        val *= multiplier.data_ptr<float>()[c];
                        
                        // Second LeakyReLU
                        val = val > 0 ? val : val * negative_slope;
                        
                        output.data_ptr<float>()[idx] = val;
                    }
                }
            }
        }
    }

    // Max Pooling (kernel_size=2)
    return at::max_pool3d(output, {2, 2, 2});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward pass for module_fn (CPU)");
}
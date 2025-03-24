#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_leaky_relu_multiply_kernel(
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
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < W && y < H && z < D) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                const int idx = ((n * C + c) * D + z) * H * W + y * W + x;
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

    // Configure thread block and grid dimensions
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(
        (W + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (H + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (D + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    fused_leaky_relu_multiply_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Forward pass for module_fn (CUDA)");
}
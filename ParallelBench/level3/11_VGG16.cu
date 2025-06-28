#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Kernel for 2D convolution with atomic operations optimization
__global__ void conv2d_atomic_kernel(const float* input, const float* weight, const float* bias,
                                      float* output, int N, int C, int H, int W,
                                      int K, int P, int stride) {
    extern __shared__ float shared_mem[];
    float* tile = shared_mem;

    int n = blockIdx.z;
    int k = blockIdx.y;
    int h = blockIdx.x * TILE_SIZE + threadIdx.y;
    int w = threadIdx.x;

    float sum = 0.0f;
    if (k < K && h < H && w < W) {
        sum = bias[k];

        #pragma unroll
        for (int c = 0; c < C; ++c) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
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
    }
    __syncthreads();

    if (k < K && h < H && w < W) {
        atomicAdd(&output[n * K * H * W + k * H * W + h * W + w], sum);
    }
}

// Optimized VGG16 forward pass using the custom optimized convolution
torch::Tensor optimized_vgg16_forward_cuda(
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

    dim3 block(TILE_SIZE, BLOCK_SIZE / TILE_SIZE);
    dim3 grid((W + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE, N);

    size_t shared_mem_size = TILE_SIZE * TILE_SIZE * sizeof(float);
    conv2d_atomic_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        conv_weights[0].data_ptr<float>(),
        conv_biases[0].data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, K, P, 1
    );

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
    m.def("forward", &optimized_vgg16_forward_cuda, "Optimized VGG16 forward (CUDA)");
}
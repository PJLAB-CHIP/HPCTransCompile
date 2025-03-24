#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 512  

template<int KernelSize>
__global__ void optimized_avgpool_kernel(const float* __restrict__ input, float* __restrict__ output,
                                        int N, int C, int pooled_D, int pooled_H, int pooled_W,
                                        int input_D, int input_H, int input_W) {
    const int total = N * C * pooled_D * pooled_H * pooled_W;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < total; 
         i += gridDim.x * blockDim.x) {
        
        const int pw = i % pooled_W;
        const int ph = (i / pooled_W) % pooled_H;
        const int pd = (i / (pooled_W * pooled_H)) % pooled_D;
        const int c  = (i / (pooled_W * pooled_H * pooled_D)) % C;
        const int n  = i / (pooled_W * pooled_H * pooled_D * C);

        const int d_start = pd * KernelSize;
        const int h_start = ph * KernelSize;
        const int w_start = pw * KernelSize;

        float sum = 0.0f;
        
        #pragma unroll
        for (int dz = 0; dz < KernelSize; ++dz) {
            const int d = d_start + dz;
            #pragma unroll
            for (int dy = 0; dy < KernelSize; ++dy) {
                const int h = h_start + dy;
                #pragma unroll
                for (int dx = 0; dx < KernelSize; ++dx) {
                    const int w = w_start + dx;
                    sum += input[((n * C + c) * input_D + d) * (input_H * input_W)
                               + h * input_W + w];
                }
            }
        }

        output[i] = sum / (KernelSize*KernelSize*KernelSize);
    }
}

at::Tensor module_fn_forward(
    at::Tensor x,
    int64_t stride,
    int64_t padding,
    at::Tensor conv_transpose,
    at::Tensor conv_transpose_bias,
    at::Tensor bn_weight,
    at::Tensor bn_bias,
    at::Tensor bn_running_mean,
    at::Tensor bn_running_var,
    at::Tensor bn_eps,
    at::Tensor bn_momentum
) {
    // Input validation checks (unchanged from previous implementation)
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    // ... (other tensor checks)

    // Existing convolution + batch norm
    auto y = at::conv_transpose3d(x, conv_transpose, conv_transpose_bias, 
                                 {stride, stride, stride}, {padding, padding, padding});
    y = at::batch_norm(y, bn_weight, bn_bias, bn_running_mean, bn_running_var, 
                      true, bn_momentum.item<double>(), bn_eps.item<double>(), true);

    // Prepare for fused kernel
    const auto sizes = y.sizes();
    const int pooled_D = sizes[2]/4, pooled_H = sizes[3]/4, pooled_W = sizes[4]/4;
    auto output = at::empty({sizes[0], sizes[1], pooled_D, pooled_H, pooled_W}, y.options());

    // Launch config with optimal block/grid sizing
    const int blocks = (output.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    optimized_avgpool_kernel<4><<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<float>(), output.data_ptr<float>(),
        sizes[0], sizes[1],
        pooled_D, pooled_H, pooled_W,
        sizes[2], sizes[3], sizes[4]
    );

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Warp-uniform fused pool kernel");
}
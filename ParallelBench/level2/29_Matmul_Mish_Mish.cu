#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float softplus(float x) {
    float abs_x = fabsf(x);
    float z = expf(-abs_x);
    return fmaxf(x, 0.0f) + log1pf(z);
}

__device__ float mish(float x) {
    float sp = softplus(x);
    return x * tanhf(sp);
}

__global__ void forward_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int i = idx / out_features;
    int j = idx % out_features;

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += x[i * in_features + k] * weight[j * in_features + k];
    }
    sum += bias[j];

    float y = mish(sum);
    y = mish(y);
    output[idx] = y;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "weight shape mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "bias shape mismatch");

    auto output = torch::empty({batch_size, out_features}, x.options());

    int total_elements = batch_size * out_features;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    forward_kernel<<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Linear double Mish forward (CUDA)");
}
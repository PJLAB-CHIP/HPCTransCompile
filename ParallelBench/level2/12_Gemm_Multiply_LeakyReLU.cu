#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __inline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void module_fn_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float multiplier,
    const float negative_slope
) {
    const int row = blockIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x;
    
    if (row >= batch_size || col >= out_features) return;

    const float* x_row = x + row * in_features;
    const float* weight_col = weight + col * in_features;
    
    float thread_sum = 0.0f;
    for (int k = lane_id; k < in_features; k += 32) {
        thread_sum += x_row[k] * weight_col[k];
    }
    
    float sum = warp_reduce_sum(thread_sum);
    
    if (lane_id == 0) {
        sum += bias[col];
        sum *= multiplier;
        output[row * out_features + col] = sum > 0 ? sum : sum * negative_slope;
    }
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    float multiplier,
    float negative_slope,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.device().is_cuda(), "bias must be a CUDA tensor");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "Weight in_features must match x in_features");
    TORCH_CHECK(bias.size(0) == out_features, "Bias size must match weight out_features");

    auto output = torch::zeros({batch_size, out_features}, x.options());

    dim3 block(32, 16);
    dim3 grid(
        batch_size,
        (out_features + block.y - 1) / block.y
    );

    module_fn_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        multiplier,
        negative_slope
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Module function forward CUDA with warp primitives");
}
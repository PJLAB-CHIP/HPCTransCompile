#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

__global__ void module_fn_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float scaling_factor)
{
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch_size && col < out_features) {
        float val = 0.0f;
        
        // Each warp handles a portion of the reduction
        for (int k = lane_id; k < in_features; k += warp_size) {
            val += x[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        // First thread in warp has final sum
        if (lane_id == 0) {
            val += bias[col];
            float original_val = val;
            val *= scaling_factor;
            val += original_val;
            out[row * out_features + col] = val;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    const float scaling_factor,
    torch::Tensor weight,
    torch::Tensor bias)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    
    auto x_ = x.contiguous();
    auto w_ = weight.contiguous();
    auto b_ = bias.contiguous();

    const int batch_size = x_.size(0);
    const int in_features = x_.size(1);
    const int out_features = w_.size(0);

    auto out = torch::empty({batch_size, out_features}, x_.options());

    // Configure block and grid sizes optimized for warp operations
    dim3 block(32, 16); // 32 threads per warp
    dim3 grid(batch_size, (out_features + block.y - 1) / block.y);

    module_fn_forward_kernel<<<grid, block>>>(
        x_.data_ptr<float>(),
        w_.data_ptr<float>(),
        b_.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor
    );

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "module_fn forward (CUDA)");
}
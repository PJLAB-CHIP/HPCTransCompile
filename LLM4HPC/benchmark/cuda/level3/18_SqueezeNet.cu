#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cuda_runtime.h>

namespace py = pybind11;

template<int BLOCK_SIZE = 256>
__global__ void adaptive_avg_pool2d_shared_kernel(const float* __restrict__ input,
                                                float* __restrict__ output,
                                                int N, int C, int H, int W) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    const int idx = blockIdx.x;  // Each block handles one (n,c) pair
    if (idx >= N * C) return;

    const int tid = threadIdx.x;
    const int n = idx / C;
    const int c = idx % C;
    const int total = H * W;
    
    // First reduction step using shared memory
    float thread_sum = 0.0f;
    for (int i = tid; i < total; i += BLOCK_SIZE) {
        const int h = i / W;
        const int w = i % W;
        const int offset = ((n * C + c) * H + h) * W + w;
        thread_sum += input[offset];
    }
    
    // Store in shared memory
    shared_data[tid] = thread_sum;
    __syncthreads();

    // Reduce within block using shared memory
    #pragma unroll
    for (int s = BLOCK_SIZE/2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Final reduction within the first warp
    if (tid < 32) {
        float warp_sum = shared_data[tid];
        if (BLOCK_SIZE > 32) warp_sum += shared_data[tid + 32];
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        if (tid == 0) {
            output[idx] = warp_sum / static_cast<float>(total);
        }
    }
}

__global__ void relu_kernel(float* data, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        const float val = data[i];
        data[i] = (val > 0.f) ? val : 0.f;
    }
}

torch::Tensor custom_relu(torch::Tensor input) {
    const int threads = 256;
    const int n = input.numel();
    const int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input.data_ptr<float>(), n);
    return input;
}

torch::Tensor forward(torch::Tensor x, py::object params_obj) {
    using namespace torch;
    
    std::map<std::string, Tensor> params;
    py::dict params_dict = params_obj.attr("items")();
    for (auto item : params_dict) {
        std::string key = py::cast<std::string>(item.first);
        Tensor value = py::cast<Tensor>(item.second);
        params[key] = value.contiguous();
    }

    if (!x.is_contiguous()) x = x.contiguous();
    
    x = conv2d(x, params["conv1_weight"], params["conv1_bias"],
               /*stride=*/at::IntArrayRef{2, 2},
               /*padding=*/at::IntArrayRef{0, 0},
               /*dilation=*/at::IntArrayRef{1, 1},
               /*groups=*/1);
    x = custom_relu(x);
    x = max_pool2d(x,
                   /*kernel_size=*/at::IntArrayRef{3, 3},
                   /*stride=*/at::IntArrayRef{2, 2},
                   /*padding=*/at::IntArrayRef{0, 0},
                   /*dilation=*/at::IntArrayRef{1, 1},
                   /*ceil_mode=*/true);

    auto fire_module = [&params](Tensor x, std::string prefix) {
        x = x.contiguous();
        Tensor squeeze = conv2d(x,
                              params[prefix + "_squeeze_weight"],
                              params[prefix + "_squeeze_bias"],
                              /*stride=*/at::IntArrayRef{1, 1},
                              /*padding=*/at::IntArrayRef{0, 0},
                              /*dilation=*/at::IntArrayRef{1, 1},
                              /*groups=*/1);
        squeeze = custom_relu(squeeze);

        squeeze = squeeze.contiguous();
        Tensor e1 = conv2d(squeeze,
                         params[prefix + "_expand1x1_weight"],
                         params[prefix + "_expand1x1_bias"],
                         /*stride=*/at::IntArrayRef{1, 1},
                         /*padding=*/at::IntArrayRef{0, 0},
                         /*dilation=*/at::IntArrayRef{1, 1},
                         /*groups=*/1);
        e1 = custom_relu(e1);
        
        Tensor e3 = conv2d(squeeze,
                         params[prefix + "_expand3x3_weight"],
                         params[prefix + "_expand3x3_bias"],
                         /*stride=*/at::IntArrayRef{1, 1},
                         /*padding=*/at::IntArrayRef{1, 1},
                         /*dilation=*/at::IntArrayRef{1, 1},
                         /*groups=*/1);
        e3 = custom_relu(e3);

        return cat({e1.contiguous(), e3.contiguous()}, /*dim=*/1);
    };

    x = fire_module(x, "fire1");
    x = fire_module(x, "fire2");
    x = fire_module(x, "fire3");
    x = max_pool2d(x,
                   /*kernel_size=*/at::IntArrayRef{3, 3},
                   /*stride=*/at::IntArrayRef{2, 2},
                   /*padding=*/at::IntArrayRef{0, 0},
                   /*dilation=*/at::IntArrayRef{1, 1},
                   /*ceil_mode=*/true);
    
    x = fire_module(x, "fire4");
    x = fire_module(x, "fire5");
    x = fire_module(x, "fire6");
    x = fire_module(x, "fire7");
    x = max_pool2d(x,
                   /*kernel_size=*/at::IntArrayRef{3, 3},
                   /*stride=*/at::IntArrayRef{2, 2},
                   /*padding=*/at::IntArrayRef{0, 0},
                   /*dilation=*/at::IntArrayRef{1, 1},
                   /*ceil_mode=*/true);
    
    x = fire_module(x, "fire8");

    x = x.contiguous();
    x = conv2d(x,
               params["classifier_weight"],
               params["classifier_bias"],
               /*stride=*/at::IntArrayRef{1, 1},
               /*padding=*/at::IntArrayRef{0, 0},
               /*dilation=*/at::IntArrayRef{1, 1},
               /*groups=*/1);
    x = custom_relu(x);
    
    auto sizes = x.sizes();
    auto out = at::empty({sizes[0], sizes[1], 1, 1}, x.options());
    
    const int pool_blocks = sizes[0] * sizes[1];  // N * C
    const int pool_threads = 256;
    adaptive_avg_pool2d_shared_kernel<256><<<pool_blocks, pool_threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        sizes[0], sizes[1], sizes[2], sizes[3]);
    
    return flatten(out, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SqueezeNet forward with shared memory reduction");
}
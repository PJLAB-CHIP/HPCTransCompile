#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

template <typename scalar_t>
__global__ void optimized_hybrid_conv3d_kernel(
    const scalar_t* __restrict__ output,
    const scalar_t* __restrict__ scaling_factor,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ result,
    const int batch_size,
    const int out_channels,
    const int depth,
    const int height,
    const int width) {

    const int total_elements = batch_size * out_channels * depth * height * width;
    const int whd = width * height * depth;
    
    extern __shared__ char smem[];
    scalar_t* s_scaling = reinterpret_cast<scalar_t*>(smem);
    scalar_t* s_bias = s_scaling + out_channels;

    for (int i = threadIdx.x; i < out_channels; i += blockDim.x) {
        s_scaling[i] = scaling_factor[i];
        s_bias[i] = bias[i];
    }
    __syncthreads();

    if constexpr (std::is_same<scalar_t, float>::value) {
        const int vec_size = 4;
        const int vec_total = total_elements / vec_size;
        
        auto output_vec = reinterpret_cast<const float4*>(output);
        auto result_vec = reinterpret_cast<float4*>(result);
        
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        #pragma unroll
        for (int i = tid; i < vec_total; i += stride) {
            float4 in_vec = __ldg(&output_vec[i]);
            float4 out_vec;
            
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int idx = i * 4 + j;
                int c_idx = (idx / whd) % out_channels;
                float val = ((float*)&in_vec)[j];
                
                val *= s_scaling[c_idx];
                val = __tanf(val);
                val *= s_bias[c_idx];
                val = __frcp_rn(1.0f + __expf(-val));
                
                ((float*)&out_vec)[j] = val;
            }
            result_vec[i] = out_vec;
        }

        for (int idx = vec_total * 4 + tid; idx < total_elements; idx += stride) {
            int c_idx = (idx / whd) % out_channels;
            float val = output[idx];
            val *= s_scaling[c_idx];
            val = __tanf(val);
            val *= s_bias[c_idx];
            val = __frcp_rn(1.0f + __expf(-val));
            result[idx] = val;
        }
    } else {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        
        for (int idx = tid; idx < total_elements; idx += stride) {
            int c_idx = (idx / whd) % out_channels;
            scalar_t val = output[idx];
            val *= s_scaling[c_idx];
            val = tanh(val);
            val *= s_bias[c_idx];
            val = scalar_t(1) / (scalar_t(1) + exp(-val));
            result[idx] = val;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor scaling_factor,
    torch::Tensor bias) {

    auto conv_out = torch::conv3d(x, conv_weight, conv_bias);
    
    const int batch_size = conv_out.size(0);
    const int out_channels = conv_out.size(1);
    const int depth = conv_out.size(2);
    const int height = conv_out.size(3);
    const int width = conv_out.size(4);
    
    auto result = torch::empty_like(conv_out);
    
    const int threads = 256;
    const int total_elements = batch_size * out_channels * depth * height * width;
    const int blocks = std::min(65535, (total_elements + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(conv_out.scalar_type(), "optimized_hybrid_conv3d_kernel", ([&] {
        optimized_hybrid_conv3d_kernel<scalar_t><<<blocks, threads, 2 * out_channels * sizeof(scalar_t)>>>(
            conv_out.data_ptr<scalar_t>(),
            scaling_factor.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            batch_size,
            out_channels,
            depth,
            height,
            width
        );
    }));

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized hybrid Conv3d forward");
}
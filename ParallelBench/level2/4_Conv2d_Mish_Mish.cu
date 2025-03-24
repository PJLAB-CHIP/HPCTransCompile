#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

__device__ inline float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__device__ inline double mish_activation(double x) {
    return x * tanh(log(1.0 + exp(x)));
}

template <typename scalar_t, bool IS_3x3=true>
__global__ void conv2d_mish_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int k_h,
    const int k_w,
    const int out_h,
    const int out_w) {

    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z % out_channels;
    const int b = blockIdx.z / out_channels;

    if (oh >= out_h || ow >= out_w || b >= batch_size) return;

    const int out_idx = b * (out_channels * out_h * out_w) +
                       oc * (out_h * out_w) +
                       oh * out_w +
                       ow;

    scalar_t sum = bias[oc];

    if constexpr (IS_3x3) {
        const int batch_offset = b * (in_channels * in_h * in_w);
        const int weight_oc_offset = oc * (in_channels * 9);

        #pragma unroll
        for (int ic = 0; ic < in_channels; ++ic) {
            const scalar_t* in_ptr = input + batch_offset + ic * in_h * in_w + oh * in_w + ow;
            const scalar_t* w_ptr = weight + weight_oc_offset + ic * 9;

            scalar_t in_vals[9];
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                in_vals[i*3 + 0] = in_ptr[i*in_w + 0];
                in_vals[i*3 + 1] = in_ptr[i*in_w + 1];
                in_vals[i*3 + 2] = in_ptr[i*in_w + 2];
            }

            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                sum += in_vals[i] * w_ptr[i];
            }
        }
    } else {
        const int batch_offset = b * (in_channels * in_h * in_w);
        const int weight_oc_offset = oc * (in_channels * k_h * k_w);

        #pragma unroll 4
        for (int ic = 0; ic < in_channels; ++ic) {
            const int in_ch_offset = batch_offset + ic * in_h * in_w;
            const int weight_ic_offset = weight_oc_offset + ic * k_h * k_w;

            #pragma unroll
            for (int kh = 0; kh < k_h; ++kh) {
                const scalar_t* in_row = input + in_ch_offset + (oh + kh) * in_w + ow;
                const scalar_t* w_row = weight + weight_ic_offset + kh * k_w;

                #pragma unroll
                for (int kw = 0; kw < k_w; ++kw) {
                    sum += in_row[kw] * w_row[kw];
                }
            }
        }
    }

    output[out_idx] = mish_activation(mish_activation(sum));
}

at::Tensor forward(at::Tensor input, at::Tensor conv_weight, at::Tensor conv_bias) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_h = input.size(2);
    const auto in_w = input.size(3);
    const auto out_channels = conv_weight.size(0);
    const auto k_h = conv_weight.size(2);
    const auto k_w = conv_weight.size(3);
    const auto out_h = in_h - k_h + 1;
    const auto out_w = in_w - k_w + 1;

    auto output = at::empty({batch_size, out_channels, out_h, out_w}, input.options());

    dim3 threads, blocks;
    if (k_h == 3 && k_w == 3) {
        threads = dim3(32, 8);
        blocks = dim3(
            (out_w + threads.x - 1) / threads.x,
            (out_h + threads.y - 1) / threads.y,
            batch_size * out_channels
        );
    } else {
        const int block_size = (out_w * out_h < 256) ? 128 : 
                             (out_w * out_h < 1024) ? 256 : 512;
        threads = dim3(block_size);
        const int total_threads = batch_size * out_channels * out_h * out_w;
        blocks = dim3((total_threads + block_size - 1) / block_size);
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv2d_mish_forward_cuda", ([&] {
        if (k_h == 3 && k_w == 3) {
            conv2d_mish_kernel<scalar_t, true><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                conv_weight.data_ptr<scalar_t>(),
                conv_bias.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                in_channels,
                in_h,
                in_w,
                out_channels,
                k_h,
                k_w,
                out_h,
                out_w);
        } else {
            conv2d_mish_kernel<scalar_t, false><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                conv_weight.data_ptr<scalar_t>(),
                conv_bias.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                in_channels,
                in_h,
                in_w,
                out_channels,
                k_h,
                k_w,
                out_h,
                out_w);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Convolution with double Mish activation (CUDA)");
}
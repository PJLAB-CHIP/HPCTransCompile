#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with balanced workload distribution
// Distribute workloads evenly across threads and blocks

template <typename scalar_t>
__global__ void process_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int pool_size,
    const int pool_out_h,
    const int pool_out_w,
    const float subtract1,
    const float subtract2
) {
    // Calculate global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * pool_out_h * pool_out_w;
    if (idx >= total_elements) return;

    // Calculate pooling indices
    const int pw = idx % pool_out_w;
    const int ph = (idx / pool_out_w) % pool_out_h;
    const int c = (idx / (pool_out_w * pool_out_h)) % out_channels;
    const int b = idx / (pool_out_w * pool_out_h * out_channels);

    // Pooling window start positions
    const int h_start = ph * pool_size;
    const int w_start = pw * pool_size;

    float pool_sum = 0.0f;
    int pool_count = 0;

    // Unroll pooling window loops
    #pragma unroll 4
    for (int ph_offset = 0; ph_offset < pool_size; ph_offset++) {
        #pragma unroll 4
        for (int pw_offset = 0; pw_offset < pool_size; pw_offset++) {
            const int h = h_start + ph_offset;
            const int w = w_start + pw_offset;
            if (h >= out_height || w >= out_width) continue;

            float conv_result = bias[c];

            // Unroll convolution loop
            #pragma unroll 4
            for (int ic = 0; ic < in_channels; ic++) {
                #pragma unroll 4
                for (int kh = 0; kh < kernel_size; kh++) {
                    #pragma unroll 4
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int in_h = h + kh;
                        const int in_w = w + kw;
                        if (in_h < in_height && in_w < in_width) {
                            const int in_idx = b * (in_channels * in_height * in_width) +
                                               ic * (in_height * in_width) +
                                               in_h * in_width + in_w;
                            const int w_idx = c * (in_channels * kernel_size * kernel_size) +
                                              ic * (kernel_size * kernel_size) +
                                              kh * kernel_size + kw;
                            conv_result += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }

            // Apply subtract, tanh, and second subtract
            conv_result = tanhf(conv_result - subtract1);
            conv_result = conv_result - subtract2;

            pool_sum += conv_result;
            pool_count++;
        }
    }

    // Write final average-pooled result
    if (pool_count > 0) {
        output[idx] = pool_sum / pool_count;
    }
}

// Host function invoking the CUDA kernel

torch::Tensor forward(
    torch::Tensor input,
    int kernel_size_pool,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    float subtract1_value,
    float subtract2_value
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = conv_weight.size(0);
    const int kernel_size = conv_weight.size(2);
    
    const int out_height = in_height - kernel_size + 1;
    const int out_width = in_width - kernel_size + 1;
    const int pool_out_h = out_height / kernel_size_pool;
    const int pool_out_w = out_width / kernel_size_pool;

    auto output = torch::zeros({batch_size, out_channels, pool_out_h, pool_out_w}, input.options());

    const int total_elements = batch_size * out_channels * pool_out_h * pool_out_w;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "process_kernel", ([&] {
        process_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            conv_weight.data_ptr<scalar_t>(),
            conv_bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_size,
            kernel_size_pool,
            pool_out_h,
            pool_out_w,
            subtract1_value,
            subtract2_value
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced Conv+sub+tanh+sub+pool forward");
}

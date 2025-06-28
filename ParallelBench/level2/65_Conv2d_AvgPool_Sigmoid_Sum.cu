#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 512
#define POOL_SIZE 2

__global__ void conv_pool_sigmoid_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, 
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size
) {
    __shared__ float shared_mem[BLOCK_SIZE/32]; // Reduced shared memory footprint

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    if (bid >= batch_size) return;

    const int out_h = height - kernel_size + 1;
    const int out_w = width - kernel_size + 1;
    const int pool_h = out_h / POOL_SIZE;
    const int pool_w = out_w / POOL_SIZE;
    const int total_work = out_channels * pool_h * pool_w;

    const float pool_scale = 1.0f / (POOL_SIZE * POOL_SIZE);
    float thread_sum = 0.0f;

    // Increased parallelism with larger block size
    for (int idx = tid; idx < total_work; idx += BLOCK_SIZE) {
        const int oc = idx / (pool_h * pool_w);
        const int ph = idx % (pool_h * pool_w);
        const int pool_row = (ph / pool_w) * POOL_SIZE;
        const int pool_col = (ph % pool_w) * POOL_SIZE;
        
        float conv_val = bias[oc];

        #pragma unroll 8
        for (int ic = 0; ic < in_channels; ++ic) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                const int h_in = pool_row + kh;
                const float* input_row = &input[((bid * in_channels + ic) * height + h_in) * width];
                const float* weight_row = &weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size];
                
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    conv_val = __fmaf_rn(input_row[pool_col + kw], weight_row[kw], conv_val);
                }
            }
        }

        conv_val *= pool_scale;        
        thread_sum += __fdividef(1.0f, (1.0f + __expf(-conv_val)));
    }

    // Efficient 512-thread reduction hierarchy
    for (int offset = 16; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    // Warp leaders store to shared memory
    if (tid % 32 == 0)
        shared_mem[tid/32] = thread_sum;

    __syncthreads();

    // Final reduction across warps
    if (tid < 32) {
        thread_sum = tid < (BLOCK_SIZE/32) ? shared_mem[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

        if (tid == 0)
            output[bid] = thread_sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    auto output = torch::empty({batch_size}, input.options());

    conv_pool_sigmoid_sum_kernel<<<batch_size, BLOCK_SIZE, (BLOCK_SIZE/32)*sizeof(float)>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "BLOCK512 Conv+Pool+Sigmoid+Sum");
}
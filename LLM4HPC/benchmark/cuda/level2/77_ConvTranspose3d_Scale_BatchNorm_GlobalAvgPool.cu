#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void global_avg_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int spatial_size
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid * spatial_size + tid;
    
    // Use float4 for vectorized loads when possible
    float sum = 0.0f;
    float4 in4;
    
    // Vector loads for aligned elements
    int vector_size = spatial_size / 4 * 4;
    for (int i = tid * 4; i < vector_size; i += blockDim.x * 4) {
        in4 = *reinterpret_cast<const float4*>(&input[bid * spatial_size + i]);
        sum = __fmaf_rn(1.0f, in4.x, sum);
        sum = __fmaf_rn(1.0f, in4.y, sum);
        sum = __fmaf_rn(1.0f, in4.z, sum);
        sum = __fmaf_rn(1.0f, in4.w, sum);
    }
    
    // Handle remaining elements
    for (int i = vector_size + tid; i < spatial_size; i += blockDim.x) {
        sum = __fmaf_rn(1.0f, __ldg(&input[bid * spatial_size + i]), sum);
    }
    
    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce within block using sequential addressing
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = __fmaf_rn(1.0f, sdata[tid + s], sdata[tid]);
        }
        __syncthreads();
    }
    
    // Final reduction within warp
    if (tid < 32) {
        sum = sdata[tid];
        if (blockDim.x >= 64) sum = __fmaf_rn(1.0f, sdata[tid + 32], sum);
        sum = warpReduceSum(sum);
    }
    
    // Write result using fast reciprocal
    if (tid == 0) {
        output[bid] = __fmul_rn(sum, __frcp_rn((float)spatial_size));
    }
}

torch::Tensor module_fn_cuda(
    torch::Tensor x,
    double eps,
    double momentum,
    double scale_factor,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var
) {
    // Perform ConvTranspose3d
    x = torch::conv_transpose3d(
        x,
        conv_transpose,
        conv_transpose_bias,
        /*stride=*/{1, 1, 1},
        /*padding=*/{0, 0, 0},
        /*output_padding=*/{0, 0, 0},
        /*groups=*/1,
        /*dilation=*/{1, 1, 1}
    );

    // Multiply by scale_factor
    x = x * scale_factor;

    // Batch Normalization
    x = torch::batch_norm(
        x,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        /*training=*/true,
        momentum,
        eps,
        /*cudnn_enabled=*/true
    );

    // Custom global average pooling implementation
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int spatial_size = sizes[2] * sizes[3] * sizes[4];
    
    auto x_reshaped = x.view({batch_size * channels, spatial_size});
    auto output = torch::empty({batch_size * channels}, x.options());
    
    dim3 threads(512);
    dim3 blocks(batch_size * channels);
    int shared_mem_size = threads.x * sizeof(float);
    
    global_avg_pool_kernel<<<blocks, threads, shared_mem_size>>>(
        x_reshaped.data_ptr<float>(),
        output.data_ptr<float>(),
        spatial_size
    );
    
    return output.view({batch_size, channels, 1, 1, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cuda, "Module function forward (CUDA)");
}
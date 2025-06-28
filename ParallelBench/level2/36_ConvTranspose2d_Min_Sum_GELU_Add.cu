#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_SIZE 32

__global__ void fused_min_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W
) {
    __shared__ float shared_min[TILE_SIZE][TILE_SIZE];
    __shared__ float warp_sums[BLOCK_SIZE/WARP_SIZE];
    
    int n = blockIdx.x;
    int tile_h = (blockIdx.y * TILE_SIZE);
    int tile_w = (blockIdx.z * TILE_SIZE);
    int h = tile_h + threadIdx.y;
    int w = tile_w + threadIdx.x;
    
    float min_val = FLT_MAX;
    if (n < N && h < H && w < W) {
        for (int c = 0; c < C; ++c) {
            float val = input[((n * C + c) * H + h) * W + w];
            min_val = min(min_val, val);
        }
        shared_min[threadIdx.y][threadIdx.x] = min_val;
    }
    __syncthreads();
    
    if (threadIdx.y == 0 && n < N && w < W) {
        float sum = 0.0f;
        for (int th = 0; th < TILE_SIZE && (tile_h + th) < H; ++th) {
            sum += shared_min[th][threadIdx.x];
        }
        
        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        
        int warpId = threadIdx.x / WARP_SIZE;
        if ((threadIdx.x & (WARP_SIZE-1)) == 0) {
            warp_sums[warpId] = sum;
        }
    }
    __syncthreads();
    
    if (threadIdx.x < (BLOCK_SIZE/WARP_SIZE) && threadIdx.y == 0 && n < N && w < W) {
        float final_sum = warp_sums[threadIdx.x];
        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = (BLOCK_SIZE/WARP_SIZE)/2; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(mask, final_sum, offset);
        }
        
        if (threadIdx.x == 0) {
            output[(n * W) + tile_w + threadIdx.x] = final_sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    x = x.contiguous();
    x = at::conv_transpose2d(x, conv_transpose, conv_transpose_bias,
        {stride, stride}, {padding, padding}, {output_padding, output_padding}, 1, {1, 1});
    
    auto sizes = x.sizes();
    int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::zeros({N, 1, 1, W}, options);
    
    dim3 grid(N, (H + TILE_SIZE - 1) / TILE_SIZE, (W + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    fused_min_sum_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    output = at::gelu(output);
    output = output + bias;
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused min-sum reduction forward");
}
/*
Combined CUDA kernel that fuses warp-level tiling with vectorized memory accesses and loop unrolling
from two different implementations. Each warp processes a tile of output features (TILE_SIZE outputs)
for a given batch sample. The kernel uses __ldg() for read-only loads and vectorized float4 loads
for 128-bit aligned accesses. Remaining elements are unrolled as in the second kernel. 
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define WARP_SIZE 32
#define TILE_SIZE 4  // Each warp processes TILE_SIZE output elements

// Combined kernel: each block handles one or more batch samples along grid.x, and groups of output features along grid.y.
// In each block, warps are assigned a contiguous tile of TILE_SIZE outputs. Within each tile, threads cooperatively
// compute the dot product using vectorized loads and handle any remainder elements via loop unrolling.

__global__ void combined_warp_tile_kernel(const float* __restrict__ x,
                                           const float* __restrict__ weight,
                                           const float* __restrict__ bias,
                                           float* __restrict__ out,
                                           int in_features,
                                           int out_features) {
    // Each block processes one batch sample (grid.x) and a group of output features (grid.y).
    int batch_idx = blockIdx.x;

    // Calculate warp and lane indices
    int warps_per_block = blockDim.x / WARP_SIZE;  // e.g., 8 warps per block
    int warp_id = threadIdx.x / WARP_SIZE;           
    int lane_id = threadIdx.x % WARP_SIZE;

    // Compute the starting output index for this warp's tile
    int base_out_group = blockIdx.y * (warps_per_block * TILE_SIZE);
    int out_base = base_out_group + warp_id * TILE_SIZE;

    // Early exit if the starting output index is out-of-bounds
    if (out_base >= out_features) return;

    // Pointer to the current batch row
    const float* x_row = x + batch_idx * in_features;

    // Array to hold partial sums for each output within the tile
    float sums[TILE_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Determine vectorized loop parameters
    int nvec = in_features / 4;  // number of float4 loads
    int rem = in_features % 4;   // remaining elements

    // Process each output feature in the tile
    #pragma unroll
    for (int tile = 0; tile < TILE_SIZE; tile++) {
        int current_out = out_base + tile;
        if (current_out < out_features) {
            const float* w_row = weight + current_out * in_features;

            // Cast pointers for vectorized loads (assumes data is 128-bit aligned)
            const float4* x_vec = reinterpret_cast<const float4*>(x_row);
            const float4* w_vec = reinterpret_cast<const float4*>(w_row);

            float sum = 0.0f;

            // Main vectorized loop: each thread handles a stride of WARP_SIZE
            for (int k = lane_id; k < nvec; k += WARP_SIZE) {
                float4 a = __ldg(&x_vec[k]);
                float4 b = __ldg(&w_vec[k]);
                sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }

            // Handle remainder elements (if any) with loop unrolling
            int offset = nvec * 4;
            for (int r = lane_id; r < rem; r += WARP_SIZE) {
                sum += __ldg(x_row + offset + r) * __ldg(w_row + offset + r);
            }

            sums[tile] = sum;
        }
    }

    // Perform warp-level reduction using shuffle instructions
    #pragma unroll
    for (int tile = 0; tile < TILE_SIZE; tile++) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sums[tile] += __shfl_down_sync(0xffffffff, sums[tile], offset);
        }
    }

    // The first lane of each warp writes the final output with bias and ReLU activation
    if (lane_id == 0) {
        for (int tile = 0; tile < TILE_SIZE; tile++) {
            int current_out = out_base + tile;
            if (current_out < out_features) {
                float result = sums[tile] + __ldg(bias + current_out);
                out[batch_idx * out_features + current_out] = (result > 0.0f) ? result : 0.0f;
            }
        }
    }
}

// Host function to launch the combined kernel
// Grid configuration: blockIdx.x corresponds to batch index; blockIdx.y covers groups of output features.
// Block configuration: fixed number of threads, with each warp handling TILE_SIZE outputs.

torch::Tensor combined_linear_relu_forward(torch::Tensor x,
                                             torch::Tensor weight,
                                             torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto out = torch::empty({batch_size, out_features}, x.options());

    // Configure execution parameters
    int warps_per_block = 8;  // can be tuned
    int threads_per_block = warps_per_block * WARP_SIZE;
    int blocks_y = (out_features + (warps_per_block * TILE_SIZE) - 1) / (warps_per_block * TILE_SIZE);

    dim3 grid(batch_size, blocks_y);
    dim3 block(threads_per_block);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    combined_warp_tile_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        in_features,
        out_features
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_linear_relu_forward, "Combined GEMM with bias and ReLU (CUDA) using warp-level tile and vectorized memory access");
}

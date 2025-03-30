#include <torch/extension.h>
#include <omp.h>
#include <vector>

#define WARP_SIZE 32
#define TILE_SIZE 4  // Each warp processes TILE_SIZE output elements

// Combined kernel: each block handles one or more batch samples along grid.x, and groups of output features along grid.y.
// In each block, warps are assigned a contiguous tile of TILE_SIZE outputs. Within each tile, threads cooperatively
// compute the dot product using vectorized loads and handle any remainder elements via loop unrolling.

torch::Tensor combined_linear_relu_forward(torch::Tensor x,
                                             torch::Tensor weight,
                                             torch::Tensor bias) {
    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(weight.is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(bias.is_cpu(), "bias must be a CPU tensor");

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

    // Launch the combined kernel
    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const float* x_row = x.data_ptr<float>() + batch_idx * in_features;
        const float* weight_row = weight.data_ptr<float>();
        const float* bias_row = bias.data_ptr<float>();
        float* out_row = out.data_ptr<float>() + batch_idx * out_features;

        for (int out_base = 0; out_base < out_features; out_base += TILE_SIZE * warps_per_block) {
            #pragma omp parallel for
            for (int warp_id = 0; warp_id < warps_per_block; ++warp_id) {
                int lane_id = omp_get_thread_num() % WARP_SIZE;
                float sums[TILE_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};

                int tile_start = out_base + warp_id * TILE_SIZE;
                int tile_end = std::min(tile_start + TILE_SIZE, out_features);

                for (int tile = tile_start; tile < tile_end; ++tile) {
                    float sum = 0.0f;
                    int current_out = tile;
                    const float* w_row = weight_row + current_out * in_features;

                    int nvec = in_features / 4;  // number of float4 loads
                    int rem = in_features % 4;   // remaining elements

                    for (int k = 0; k < nvec; ++k) {
                        float4 a = reinterpret_cast<const float4*>(x_row)[k];
                        float4 b = reinterpret_cast<const float4*>(w_row)[k];
                        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                    }

                    for (int r = 0; r < rem; ++r) {
                        sum += x_row[nvec * 4 + r] * w_row[nvec * 4 + r];
                    }

                    sums[tile - tile_start] = sum;
                }

                for (int tile = 0; tile < tile_end - tile_start; ++tile) {
                    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                        sums[tile] += sums[tile] + __shfl_down_sync(0xffffffff, sums[tile], offset);
                    }
                }

                if (lane_id == 0) {
                    for (int tile = 0; tile < tile_end - tile_start; ++tile) {
                        int current_out = tile_start + tile;
                        float result = sums[tile] + bias_row[current_out];
                        out_row[current_out] = (result > 0.0f) ? result : 0.0f;
                    }
                }
            }
        }
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_linear_relu_forward, "Combined GEMM with bias and ReLU (CPU) using warp-level tile and vectorized memory access");
}
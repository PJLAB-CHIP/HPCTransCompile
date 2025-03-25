#include <torch/extension.h>
#include <omp.h>

#define TILE_SIZE 4  // Each thread processes TILE_SIZE output elements

// Combined CPU function: each thread handles one or more batch samples along the outer dimension, and groups of output features along the inner dimension.
torch::Tensor combined_linear_relu_forward_cpu(torch::Tensor x,
                                               torch::Tensor weight,
                                               torch::Tensor bias) {
    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor");
    TORCH_CHECK(!weight.is_cuda(), "weight must be a CPU tensor");
    TORCH_CHECK(!bias.is_cuda(), "bias must be a CPU tensor");

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto out = torch::empty({batch_size, out_features}, x.options());

    #pragma omp parallel for collapse(2)
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int out_base = 0; out_base < out_features; out_base += TILE_SIZE) {
            // Array to hold partial sums for each output within the tile
            float sums[TILE_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};

            // Pointer to the current batch row
            const float* x_row = x.data_ptr<float>() + batch_idx * in_features;

            // Process each output feature in the tile
            for (int tile = 0; tile < TILE_SIZE; ++tile) {
                int current_out = out_base + tile;
                if (current_out < out_features) {
                    const float* w_row = weight.data_ptr<float>() + current_out * in_features;

                    // Main loop: each thread handles a stride of TILE_SIZE
                    for (int k = 0; k < in_features; ++k) {
                        sums[tile] += x_row[k] * w_row[k];
                    }
                }
            }

            // The first lane of each thread writes the final output with bias and ReLU activation
            for (int tile = 0; tile < TILE_SIZE; ++tile) {
                int current_out = out_base + tile;
                if (current_out < out_features) {
                    float result = sums[tile] + bias.data_ptr<float>()[current_out];
                    out.data_ptr<float>()[batch_idx * out_features + current_out] = (result > 0.0f) ? result : 0.0f;
                }
            }
        }
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_linear_relu_forward_cpu, "Combined GEMM with bias and ReLU (CPU) using tile and scalar memory access");
}
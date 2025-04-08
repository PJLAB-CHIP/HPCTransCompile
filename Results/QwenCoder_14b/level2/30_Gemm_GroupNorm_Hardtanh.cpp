#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <cmath>

#define TILE_SIZE 16

namespace py = pybind11;

// Load a tile from the input matrix (x) into a local array
template <typename scalar_t, int TILE_SIZE>
void load_tile_A(const scalar_t* x, scalar_t A_tile[TILE_SIZE][TILE_SIZE], int row, int t, int in_features) {
    int col = t * TILE_SIZE;
    for (int i = 0; i < TILE_SIZE; ++i) {
        A_tile[0][i] = (col + i < in_features) ? x[row * in_features + col + i] : static_cast<scalar_t>(0);
    }
}

// Load a tile from the weight matrix into a local array
template <typename scalar_t, int TILE_SIZE>
void load_tile_B(const scalar_t* weight, scalar_t B_tile[TILE_SIZE][TILE_SIZE], int col, int t, int in_features) {
    int k = t * TILE_SIZE;
    for (int i = 0; i < TILE_SIZE; ++i) {
        B_tile[i][0] = (k + i < in_features) ? weight[col * in_features + k + i] : static_cast<scalar_t>(0);
    }
}

// Compute dot product on a single tile loaded into local arrays
template <typename scalar_t, int TILE_SIZE>
scalar_t compute_tile_dot(scalar_t A_tile[TILE_SIZE][TILE_SIZE], scalar_t B_tile[TILE_SIZE][TILE_SIZE]) {
    scalar_t sum = 0;
    for (int i = 0; i < TILE_SIZE; ++i) {
        sum += A_tile[0][i] * B_tile[i][0];
    }
    return sum;
}

// Linear Forward Function using modular functions and tiling
void linear_forward_cpu_modular(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output) {

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    int numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < out_features; ++col) {
            scalar_t sum = 0;
            scalar_t A_tile[TILE_SIZE][TILE_SIZE];
            scalar_t B_tile[TILE_SIZE][TILE_SIZE];

            for (int t = 0; t < numTiles; ++t) {
                load_tile_A<scalar_t, TILE_SIZE>(x.data_ptr<scalar_t>(), A_tile, row, t, in_features);
                load_tile_B<scalar_t, TILE_SIZE>(weight.data_ptr<scalar_t>(), B_tile, col, t, in_features);
                sum += compute_tile_dot<scalar_t, TILE_SIZE>(A_tile, B_tile);
            }

            output[row * out_features + col] = sum + bias[col];
        }
    }
}

// Block-wise reduction to sum up values
template <typename scalar_t>
scalar_t blockReduceSum(scalar_t* sdata, int tid, int blockDim) {
    for (int s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
    }
    return sdata[0];
}

// Group Normalization Function: Each thread handles one (batch, group) pair
void group_norm_forward_cpu_modular(
    const torch::Tensor& x,
    const torch::Tensor& gamma,  // scale parameter
    const torch::Tensor& beta,   // shift parameter
    torch::Tensor& output,
    int num_groups) {

    int batch_size = x.size(0);
    int num_channels = x.size(1);
    int channels_per_group = num_channels / num_groups;

    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int group = 0; group < num_groups; ++group) {
            scalar_t sum = 0;
            for (int i = 0; i < channels_per_group; ++i) {
                int channel = group * channels_per_group + i;
                sum += x[batch * num_channels + channel];
            }

            scalar_t mean = sum / channels_per_group;

            scalar_t sq_sum = 0;
            for (int i = 0; i < channels_per_group; ++i) {
                int channel = group * channels_per_group + i;
                scalar_t diff = x[batch * num_channels + channel] - mean;
                sq_sum += diff * diff;
            }

            scalar_t var = sq_sum / channels_per_group;
            scalar_t inv_std = std::rsqrt(var + 1e-5f);

            for (int i = 0; i < channels_per_group; ++i) {
                int channel = group * channels_per_group + i;
                scalar_t val = x[batch * num_channels + channel];
                output[batch * num_channels + channel] = ((val - mean) * inv_std) * gamma[channel] + beta[channel];
            }
        }
    }
}

// Hardtanh Activation Function
template <typename scalar_t>
scalar_t hardtanh_activation(scalar_t val, scalar_t min_val, scalar_t max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

// Hardtanh Function: Applies the activation in a grid-stride loop
void hardtanh_forward_cpu_modular(
    const torch::Tensor& x,
    scalar_t min_val,
    scalar_t max_val,
    torch::Tensor& output) {

    int total_elements = x.numel();

    #pragma omp parallel for
    for (int idx = 0; idx < total_elements; ++idx) {
        scalar_t val = x[idx];
        output[idx] = hardtanh_activation<scalar_t>(val, min_val, max_val);
    }
}

// Combined Function: Executes linear, group norm, and hardtanh sequentially
torch::Tensor module_fn_cpu_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& group_norm_weight,
    const torch::Tensor& group_norm_bias,
    int64_t num_groups,
    float hardtanh_min,
    float hardtanh_max) {

    int64_t batch_size = x.size(0);
    int64_t in_features = x.size(1);
    int64_t out_features = weight.size(0);

    auto options = x.options();
    torch::Tensor linear_output = torch::empty({batch_size, out_features}, options);
    torch::Tensor group_norm_output = torch::empty({batch_size, out_features}, options);
    torch::Tensor output = torch::empty({batch_size, out_features}, options);

    // Linear layer computation with tiling
    linear_forward_cpu_modular(x, weight, bias, linear_output);

    // Group Normalization with parallel reduction per group
    group_norm_forward_cpu_modular(linear_output, group_norm_weight, group_norm_bias, group_norm_output, num_groups);

    // Hardtanh activation using a grid-stride loop
    hardtanh_forward_cpu_modular(group_norm_output, hardtanh_min, hardtanh_max, output);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu_forward, "Forward pass (CPU modular optimized)");
}

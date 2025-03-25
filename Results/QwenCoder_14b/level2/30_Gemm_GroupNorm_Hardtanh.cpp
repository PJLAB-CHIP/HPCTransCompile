#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <cmath>

constexpr int TILE_SIZE = 16;

template <typename scalar_t, int TILE_SIZE>
void load_tile_A(const scalar_t* x, scalar_t A_tile[TILE_SIZE][TILE_SIZE], int row, int t, int in_features) {
    int col = t * TILE_SIZE;
    for (int i = 0; i < TILE_SIZE; ++i) {
        A_tile[0][i] = (col + i < in_features) ? x[row * in_features + col + i] : static_cast<scalar_t>(0);
    }
}

template <typename scalar_t, int TILE_SIZE>
void load_tile_B(const scalar_t* weight, scalar_t B_tile[TILE_SIZE][TILE_SIZE], int col, int t, int in_features) {
    int k = t * TILE_SIZE;
    for (int i = 0; i < TILE_SIZE; ++i) {
        B_tile[i][0] = (k + i < in_features) ? weight[col * in_features + k + i] : static_cast<scalar_t>(0);
    }
}

template <typename scalar_t, int TILE_SIZE>
scalar_t compute_tile_dot(scalar_t A_tile[TILE_SIZE][TILE_SIZE], scalar_t B_tile[TILE_SIZE][TILE_SIZE]) {
    scalar_t sum = 0;
    for (int i = 0; i < TILE_SIZE; ++i) {
        sum += A_tile[0][i] * B_tile[i][0];
    }
    return sum;
}

template <typename scalar_t, int TILE_SIZE>
void linear_forward_cpu_modular(
    const scalar_t* x,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    int batch_size,
    int in_features,
    int out_features) {
    int numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < out_features; ++col) {
            scalar_t sum = 0;
            scalar_t A_tile[TILE_SIZE][TILE_SIZE];
            scalar_t B_tile[TILE_SIZE][TILE_SIZE];

            for (int t = 0; t < numTiles; ++t) {
                load_tile_A<scalar_t, TILE_SIZE>(x, A_tile, row, t, in_features);
                load_tile_B<scalar_t, TILE_SIZE>(weight, B_tile, col, t, in_features);
                sum += compute_tile_dot<scalar_t, TILE_SIZE>(A_tile, B_tile);
            }
            output[row * out_features + col] = sum + bias[col];
        }
    }
}

template <typename scalar_t>
scalar_t blockReduceSum(scalar_t* sdata, int tid, int blockDim) {
    for (int s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
    }
    return sdata[0];
}

template <typename scalar_t>
void group_norm_forward_cpu_modular(
    const scalar_t* x,
    const scalar_t* gamma,
    const scalar_t* beta,
    scalar_t* output,
    int batch_size,
    int num_channels,
    int num_groups) {
    int channels_per_group = num_channels / num_groups;

    #pragma omp parallel for
    for (int idx = 0; idx < batch_size * num_groups; ++idx) {
        int batch = idx / num_groups;
        int group = idx % num_groups;
        scalar_t sum = 0;

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < channels_per_group; ++i) {
            int channel = group * channels_per_group + i;
            sum += x[batch * num_channels + channel];
        }

        scalar_t mean = sum / channels_per_group;

        scalar_t sq_sum = 0;
        #pragma omp parallel for reduction(+:sq_sum)
        for (int i = 0; i < channels_per_group; ++i) {
            int channel = group * channels_per_group + i;
            scalar_t diff = x[batch * num_channels + channel] - mean;
            sq_sum += diff * diff;
        }

        scalar_t var = sq_sum / channels_per_group;
        scalar_t inv_std = std::rsqrt(var + 1e-5f);

        #pragma omp parallel for
        for (int i = 0; i < channels_per_group; ++i) {
            int channel = group * channels_per_group + i;
            scalar_t val = x[batch * num_channels + channel];
            output[batch * num_channels + channel] = ((val - mean) * inv_std) * gamma[channel] + beta[channel];
        }
    }
}

template <typename scalar_t>
void hardtanh_forward_cpu_modular(
    const scalar_t* x,
    scalar_t min_val,
    scalar_t max_val,
    scalar_t* output,
    size_t total_elements) {
    #pragma omp parallel for
    for (size_t idx = 0; idx < total_elements; ++idx) {
        scalar_t val = x[idx];
        output[idx] = (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
    }
}

void module_fn_cpu_forward(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor group_norm_weight,
    at::Tensor group_norm_bias,
    int64_t num_groups,
    float hardtanh_min,
    float hardtanh_max) {
    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    auto options = x.options();
    at::Tensor linear_output = at::empty({batch_size, out_features}, options);
    at::Tensor group_norm_output = at::empty({batch_size, out_features}, options);
    at::Tensor output = at::empty({batch_size, out_features}, options);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "linear_forward_cpu_modular", ([&] {
        linear_forward_cpu_modular<scalar_t, TILE_SIZE>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            linear_output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features);
    }));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cpu_modular", ([&] {
        group_norm_forward_cpu_modular<scalar_t>(
            linear_output.data_ptr<scalar_t>(),
            group_norm_weight.data_ptr<scalar_t>(),
            group_norm_bias.data_ptr<scalar_t>(),
            group_norm_output.data_ptr<scalar_t>(),
            batch_size,
            out_features,
            num_groups);
    }));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_forward_cpu_modular", ([&] {
        hardtanh_forward_cpu_modular<scalar_t>(
            group_norm_output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(hardtanh_min),
            static_cast<scalar_t>(hardtanh_max),
            output.data_ptr<scalar_t>(),
            x.numel());
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu_forward, "Forward pass (CPU modular optimized)");
}
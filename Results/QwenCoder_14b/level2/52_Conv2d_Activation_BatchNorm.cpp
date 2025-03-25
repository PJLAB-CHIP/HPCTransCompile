#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Helper functions for math operations

template <typename scalar_t>
inline scalar_t my_exp(scalar_t x);

template <>
inline float my_exp<float>(float x) { return expf(x); }

template <>
inline double my_exp<double>(double x) { return exp(x); }

template <typename scalar_t>
inline scalar_t my_log1p(scalar_t x);

template <>
inline float my_log1p<float>(float x) { return log1pf(x); }

template <>
inline double my_log1p<double>(double x) { return log1p(x); }

template <typename scalar_t>
inline scalar_t my_tanh(scalar_t x);

template <>
inline float my_tanh<float>(float x) { return tanhf(x); }

template <>
inline double my_tanh<double>(double x) { return tanh(x); }

// Function to perform fused activation and per-channel reduction
template <typename scalar_t>
void activation_reduction_cpu(
    const scalar_t* x,
    scalar_t* y,
    int N, int C, int H, int W,
    scalar_t* d_sum,
    scalar_t* d_sumsq) {

#pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        scalar_t local_sum = 0;
        scalar_t local_sumsq = 0;
        int count = N * H * W;
        for (int i = 0; i < count; ++i) {
            int HW = H * W;
            int n = i / HW;
            int rem = i % HW;
            int h = rem / W;
            int w = rem % W;
            int offset = n * (C * H * W) + c * (HW) + h * W + w;
            scalar_t val = x[offset];
            scalar_t sp = my_log1p<scalar_t>(my_exp<scalar_t>(val)); // softplus(x)
            scalar_t th = my_tanh<scalar_t>(sp); // tanh(softplus(x))
            scalar_t act = val * th;             // x * tanh(softplus(x))
            y[offset] = act;
            local_sum += act;
            local_sumsq += act * act;
        }
        d_sum[c] = local_sum;
        d_sumsq[c] = local_sumsq;
    }
}

// Function to perform batch normalization
template <typename scalar_t>
void batchnorm_cpu(
    scalar_t* y,
    int N, int C, int H, int W,
    const scalar_t* d_sum,
    const scalar_t* d_sumsq,
    const scalar_t* bn_weight,
    const scalar_t* bn_bias,
    scalar_t eps) {

    int total = N * C * H * W;
#pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        int w = i % W;
        int h = (i / W) % H;
        int c = (i / (W * H)) % C;
        scalar_t count = static_cast<scalar_t>(N * H * W);
        scalar_t mean = d_sum[c] / count;
        scalar_t var = d_sumsq[c] / count - mean * mean;
        scalar_t norm = (y[i] - mean) / sqrt(var + eps);
        y[i] = bn_weight[c] * norm + bn_bias[c];
    }
}

// Forward function: performs convolution, then fused activation with reduction, and finally batch normalization.
torch::Tensor forward(
    torch::Tensor x,
    double eps,
    double momentum,  // momentum is not used in fused computation; training mode is assumed
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,  // not used in fused BN
    torch::Tensor bn_running_var) {  // not used in fused BN

    // Convolution
    x = torch::conv2d(x, conv_weight, conv_bias);

    auto activated = torch::empty_like(x);

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int count = N * H * W; // Elements per channel

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto d_sum = torch::zeros({C}, options);
    auto d_sumsq = torch::zeros({C}, options);

    // Perform fused activation and reduction
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "activation_reduction_cpu", ([&] {
        activation_reduction_cpu<scalar_t>(
            x.data_ptr<scalar_t>(),
            activated.data_ptr<scalar_t>(),
            N, C, H, W,
            d_sum.data_ptr<scalar_t>(),
            d_sumsq.data_ptr<scalar_t>());
    }));

    // Perform batch normalization
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "batchnorm_cpu", ([&] {
        batchnorm_cpu<scalar_t>(
            activated.data_ptr<scalar_t>(),
            N, C, H, W,
            d_sum.data_ptr<scalar_t>(),
            d_sumsq.data_ptr<scalar_t>(),
            bn_weight.data_ptr<scalar_t>(),
            bn_bias.data_ptr<scalar_t>(),
            static_cast<scalar_t>(eps));
    }));

    return activated;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused activation and batch normalization forward (CPU)");
}
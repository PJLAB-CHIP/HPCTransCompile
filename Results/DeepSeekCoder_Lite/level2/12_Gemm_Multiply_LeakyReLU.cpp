#include <torch/extension.h>
#include <omp.h>

__inline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    float multiplier,
    float negative_slope,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(bias.device().is_cpu(), "bias must be a CPU tensor");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "Weight in_features must match x in_features");
    TORCH_CHECK(bias.size(0) == out_features, "Bias size must match weight out_features");

    auto output = torch::zeros({batch_size, out_features}, x.options());

    #pragma omp parallel for
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < out_features; ++col) {
            float thread_sum = 0.0f;
            const float* x_row = x.data_ptr<float>() + row * in_features;
            const float* weight_col = weight.data_ptr<float>() + col * in_features;

            for (int k = threadIdx.x; k < in_features; k += omp_get_max_threads()) {
                thread_sum += x_row[k] * weight_col[k];
            }

            float sum = warp_reduce_sum(thread_sum);

            if (threadIdx.x == 0) {
                sum += bias[col];
                sum *= multiplier;
                output[row * out_features + col] = sum > 0 ? sum : sum * negative_slope;
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Module function forward CPU with warp primitives");
}
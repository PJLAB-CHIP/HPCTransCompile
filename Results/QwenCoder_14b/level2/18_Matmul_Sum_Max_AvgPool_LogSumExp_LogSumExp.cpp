#include <torch/extension.h>
#include <omp.h>

// Define the CPU kernel function
template <typename scalar_t>
void cpu_sequence_ops_kernel(
    const scalar_t* x,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    const int batch_size,
    const int in_features,
    const int out_features) {

    #pragma omp parallel for collapse(2)
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int o = 0; o < out_features; ++o) {
            scalar_t sum_o = 0;
            for (int i = 0; i < in_features; ++i) {
                sum_o += x[batch_idx * in_features + i] * weight[o * in_features + i];
            }
            output[batch_idx] += bias[o] + sum_o;
        }
    }
}

// Host function to execute the CPU kernel
torch::Tensor sequence_ops_cpu_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, 1}, x.options());

    cpu_sequence_ops_kernel<scalar_t>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sequence_ops_cpu_forward, "Sequence Ops Forward (CPU)");
}
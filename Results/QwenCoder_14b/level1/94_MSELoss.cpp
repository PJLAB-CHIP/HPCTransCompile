#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <omp.h>

template <typename scalar_t>
void mse_forward_cpu(
    const scalar_t* preds,
    const scalar_t* tgts,
    double& sum_out,
    const int64_t num_elements
) {
    #pragma omp parallel for reduction(+:sum_out)
    for (int64_t i = 0; i < num_elements; ++i) {
        double diff = static_cast<double>(preds[i]) - static_cast<double>(tgts[i]);
        sum_out += diff * diff;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(!predictions.is_cuda(), "predictions must be a CPU tensor");
    TORCH_CHECK(!targets.is_cuda(), "targets must be a CPU tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    double sum_out = 0.0;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cpu", [&] {
        mse_forward_cpu<scalar_t>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            sum_out,
            num_elements
        );
    });

    auto result = torch::tensor({sum_out / static_cast<double>(num_elements)}, predictions.options());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CPU)");
}

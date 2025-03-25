#include <torch/extension.h>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void hinge_loss_cpu(const float* predictions, const float* targets, float* output, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        output[i] = std::max(0.0f, 1.0f - predictions[i] * targets[i]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_CONTIGUOUS(predictions);
    CHECK_CONTIGUOUS(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    hinge_loss_cpu(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Compute the mean of the output tensor
    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward");
}
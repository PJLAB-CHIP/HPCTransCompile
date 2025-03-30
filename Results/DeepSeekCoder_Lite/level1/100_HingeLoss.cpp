#include <torch/extension.h>
#include <omp.h>
#include <cmath>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    #pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        float pred = predictions[idx].item<float>();
        float target = targets[idx].item<float>();
        output[idx] = std::fmaxf(0.0f, 1.0f - pred * target);
    }

    // Compute the mean of the output tensor
    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward");
}
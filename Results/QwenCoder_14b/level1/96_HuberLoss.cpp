#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Function to compute Smooth L1 Loss
float smooth_l1_loss(const float* predictions, const float* targets, int n_elements) {
    float total_loss = 0.0f;

    #pragma omp parallel for reduction(+:total_loss)
    for (int i = 0; i < n_elements; ++i) {
        float diff = predictions[i] - targets[i];
        float abs_diff = std::fabs(diff);
        total_loss += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    return total_loss / n_elements;
}

// Host function wrapping the CPU implementation
torch::Tensor smooth_l1_loss_cpu(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(!predictions.device().is_cuda() && !targets.device().is_cuda(), "Inputs must be CPU tensors");

    int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    output[0] = smooth_l1_loss(predictions.data_ptr<float>(), targets.data_ptr<float>(), n_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cpu, "Smooth L1 Loss (CPU)");
}

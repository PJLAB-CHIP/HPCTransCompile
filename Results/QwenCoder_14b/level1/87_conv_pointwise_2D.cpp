#include <torch/extension.h>
#include <omp.h>

void forward_cpu(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < B; ++b) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < IC; ++ic) {
                        const int x_offset = b * IC * H * W + ic * H * W + h * W + w;
                        const int w_offset = oc * IC + ic;
                        sum += x[x_offset] * weight[w_offset];
                    }
                    output[b * OC * H * W + oc * H * W + h * W + w] = bias ? sum + bias[oc] : sum;
                }
            }
        }
    }
}

torch::Tensor forward_cpu_wrapper(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    // Input validation
    TORCH_CHECK(x.device().type() == torch::kCPU, "Inputs must be CPU tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->device().type() == torch::kCPU, "Bias must be CPU tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
    }

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    if (bias) {
        TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");
    }

    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    forward_cpu(x_ptr, w_ptr, b_ptr, out_ptr, B, IC, OC, H, W);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu_wrapper, "Pointwise 2D convolution forward (CPU)");
}

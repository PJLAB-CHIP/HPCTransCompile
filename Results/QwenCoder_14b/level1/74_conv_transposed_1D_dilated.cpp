#include <torch/extension.h>
#include <vector>
#include <omp.h>

void conv_transpose1d_cpu(
    const float* x,       // [N, C_in, L_in]
    const float* weight,  // [C_in, C_out, K_w]
    const float* bias,    // [C_out] or nullptr
    float* y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation)
{
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            for (int l_out = 0; l_out < L_out; ++l_out) {
                float value = (bias != nullptr) ? bias[c_out] : 0.0f;
                
                for (int c = 0; c < C_in; ++c) {
                    for (int k = 0; k < K_w; ++k) {
                        int l_in_nom = l_out + padding - k * dilation;
                        if (l_in_nom % stride == 0) {
                            int l_in = l_in_nom / stride;
                            if (l_in >= 0 && l_in < L_in) {
                                float x_val = x[n * C_in * L_in + c * L_in + l_in];
                                float w_val = weight[c * C_out * K_w + c_out * K_w + k];
                                value += x_val * w_val;
                            }
                        }
                    }
                }
                y[n * C_out * L_out + c_out * L_out + l_out] = value;
            }
        }
    }
}

torch::Tensor conv_transpose1d_forward(
    py::object x_obj,            // x: torch.Tensor
    py::object weight_obj,       // weight: torch.Tensor
    py::object bias_obj = py::none(),  // bias: torch.Tensor or None
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1)
{
    // Convert inputs to contiguous CPU tensors
    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();
    TORCH_CHECK(!x.is_cuda(), "Input tensor must be on CPU device");
    TORCH_CHECK(!weight.is_cuda(), "Weight tensor must be on CPU device");

    float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(!bias.is_cuda(), "Bias tensor must be on CPU device");
        bias_ptr = bias.data_ptr<float>();
    }

    // Extract tensor dimensions
    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);
    
    // Compute output length
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    // Allocate output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Call the CPU implementation
    conv_transpose1d_cpu(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K_w,
        stride, padding, dilation);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv_transpose1d_forward,
        "Conv Transpose1D forward (CPU)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1
    );
}
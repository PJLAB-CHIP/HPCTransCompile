#include <torch/extension.h>
#include <vector>
#include <omp.h>

torch::Tensor conv2d_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(input.is_cpu(), "Input tensor must be on CPU");
    TORCH_CHECK(weight.is_cpu(), "Weight tensor must be on CPU");

    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cpu(), "Bias tensor must be on CPU if provided");
    }

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    int64_t C_out = weight.size(0);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);
    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t padding_h = padding[0];
    int64_t padding_w = padding[1];
    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];

    int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;

    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }

    float* output_ptr = output.data_ptr<float>();

    #pragma omp parallel for
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c_out = 0; c_out < C_out; ++c_out) {
            for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                    float value = (bias_ptr != nullptr) ? bias_ptr[c_out] : 0.0f;
                    int group = c_out / (C_out / groups);
                    int c_in_start = group * (C_in / groups);
                    int c_in_end = c_in_start + (C_in / groups);

                    for (int64_t c_in = c_in_start; c_in < c_in_end; c_in += 4) {
                        for (int64_t k_h = 0; k_h < K_h; ++k_h) {
                            for (int64_t k_w = 0; k_w < K_w; ++k_w) {
                                int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
                                int w_in = w_out * stride_w - padding_w + k_w * dilation_w;

                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                    int base_input_idx = ((n * C_in) * H_in + h_in) * W_in + w_in;
                                    int base_weight_idx = ((c_out * (C_in / groups)) * K_h + k_h) * K_w + k_w;

                                    for (int64_t ci = 0; ci < 4 && (c_in + ci) < c_in_end; ++ci) {
                                        int input_idx = base_input_idx + (c_in + ci) * H_in * W_in;
                                        int weight_idx = base_weight_idx + (ci) * K_h * K_w;
                                        value += input_ptr[input_idx] * weight_ptr[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    output_ptr[output_idx] = value;
                }
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cpu, "Custom 2D convolution (CPU)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}
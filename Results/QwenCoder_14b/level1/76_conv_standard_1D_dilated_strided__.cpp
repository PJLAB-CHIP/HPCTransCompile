#include <torch/extension.h>
#include <omp.h>

void conv1d_cpu(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation
) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int o = 0; o < out_size; ++o) {
                int start_pos = o * stride;
                int end_pos = start_pos + (kernel_size - 1) * dilation;
                float sum = 0.0f;

                if (end_pos < in_size) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const float* x_base = x + b * (in_channels * in_size) + ic * in_size + start_pos;
                        const float* w_base = weight + oc * (in_channels * kernel_size) + ic * kernel_size;
                        
                        for (int k = 0; k < kernel_size; ++k) {
                            sum += x_base[k * dilation] * w_base[k];
                        }
                    }
                } else {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const float* x_base = x + b * (in_channels * in_size) + ic * in_size;
                        const float* w_base = weight + oc * (in_channels * kernel_size) + ic * kernel_size;
                        
                        for (int k = 0; k < kernel_size; ++k) {
                            int input_pos = start_pos + k * dilation;
                            bool valid = input_pos < in_size;
                            sum += valid * x_base[input_pos] * w_base[k];
                        }
                    }
                }

                if (bias != nullptr) {
                    sum += bias[oc];
                }
                
                output[b * (out_channels * out_size) + oc * out_size + o] = sum;
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
) {
    TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");

    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cpu(), "bias must be a CPU tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "Bias size mismatch");
    }

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    conv1d_cpu(
        x_data,
        weight_data,
        bias_data,
        output_data,
        B,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CPU) with OpenMP parallelization");
}
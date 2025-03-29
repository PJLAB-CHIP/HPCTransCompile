#include <torch/extension.h>
#include <omp.h>

void avg_pool1d_cpu(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    #pragma omp parallel for collapse(3)
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int channel = 0; channel < in_channels; ++channel) {
            for (int o = 0; o < output_length; ++o) {
                float sum = 0.0f;
                for (int k = 0; k < kernel_size; ++k) {
                    int pos_padded = o * stride + k;
                    int pos_input = pos_padded - padding;
                    
                    if (pos_input >= 0 && pos_input < input_length) {
                        int input_idx = batch * in_channels * input_length + channel * input_length + pos_input;
                        sum += input[input_idx];
                    }
                }
                int output_idx = batch * in_channels * output_length + channel * output_length + o;
                output[output_idx] = sum / kernel_size;
            }
        }
    }
}

torch::Tensor avg_pool1d_forward_cpu(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {
    
    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    avg_pool1d_cpu(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward_cpu, "1D Average Pooling forward (CPU)");
}

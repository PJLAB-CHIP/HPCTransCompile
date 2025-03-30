#include <torch/extension.h>
#include <omp.h>

torch::Tensor avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {
    
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    #pragma omp parallel for
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int channel = 0; channel < in_channels; ++channel) {
            for (int o = 0; o < output_length; ++o) {
                float sum = 0.0f;
                int input_length_padded = input_length + 2 * padding;
                for (int k = 0; k < kernel_size; ++k) {
                    int pos_padded = o * stride + k;
                    int pos_input = pos_padded - padding;
                    if (pos_input >= 0 && pos_input < input_length_padded) {
                        int input_idx = batch * in_channels * input_length_padded + channel * input_length_padded + pos_input;
                        sum += x.data_ptr<float>()[input_idx];
                    }
                }
                int output_idx = batch * in_channels * output_length + channel * output_length + o;
                output.data_ptr<float>()[output_idx] = sum / kernel_size;
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CPU)");
}
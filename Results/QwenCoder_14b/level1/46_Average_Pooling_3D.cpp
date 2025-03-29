#include <torch/extension.h>
#include <omp.h>

void avg_pool3d_forward_cpu(
    const float* input,
    float* output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int d_out = 0; d_out < out_d; ++d_out) {
                for (int h_out = 0; h_out < out_h; ++h_out) {
                    for (int w_out = 0; w_out < out_w; ++w_out) {
                        int d_start = d_out * stride - padding;
                        int h_start = h_out * stride - padding;
                        int w_start = w_out * stride - padding;

                        int d_start_clamped = std::max(d_start, 0);
                        int h_start_clamped = std::max(h_start, 0);
                        int w_start_clamped = std::max(w_start, 0);
                        int d_end_clamped = std::min(d_start + kernel_size, in_d);
                        int h_end_clamped = std::min(h_start + kernel_size, in_h);
                        int w_end_clamped = std::min(w_start + kernel_size, in_w);

                        float sum = 0.0f;
                        int pool_volume = kernel_size * kernel_size * kernel_size;

                        for (int d = d_start_clamped; d < d_end_clamped; ++d) {
                            for (int h = h_start_clamped; h < h_end_clamped; ++h) {
                                for (int w = w_start_clamped; w < w_end_clamped; ++w) {
                                    int input_idx = ((n * channels + c) * in_d + d) * in_h * in_w + h * in_w + w;
                                    sum += input[input_idx];
                                }
                            }
                        }

                        int output_idx = (((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out;
                        output[output_idx] = sum / static_cast<float>(pool_volume);
                    }
                }
            }
        }
    }
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    avg_pool3d_forward_cpu(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D Average Pooling forward (CPU)");
}

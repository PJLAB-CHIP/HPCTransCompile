#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <omp.h>

template <typename scalar_t, int KERNEL_SIZE>
void max_pool2d_tuned_kernel_cpu(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation
) {
    const int output_size = batch_size * channels * output_height * output_width;
    #pragma omp parallel for
    for (int output_idx = 0; output_idx < output_size; ++output_idx) {
        const int ow = output_idx % output_width;
        const int oh = (output_idx / output_width) % output_height;
        const int c = (output_idx / (output_width * output_height)) % channels;
        const int b = output_idx / (output_width * output_height * channels);

        const int input_batch_offset = b * (channels * input_height * input_width);
        const int input_channel_offset = c * (input_height * input_width);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        if constexpr (KERNEL_SIZE == 2) {
            const int ih_base = oh * stride - padding;
            const int iw_base = ow * stride - padding;

            if (ih_base >= 0 && ih_base < input_height && iw_base >= 0 && iw_base < input_width) {
                const int idx = input_batch_offset + input_channel_offset + ih_base * input_width + iw_base;
                max_val = std::max(max_val, input[idx]);
            }
            if (ih_base >= 0 && ih_base < input_height && iw_base + dilation >= 0 && iw_base + dilation < input_width) {
                const int idx = input_batch_offset + input_channel_offset + ih_base * input_width + (iw_base + dilation);
                max_val = std::max(max_val, input[idx]);
            }
            if (ih_base + dilation >= 0 && ih_base + dilation < input_height && iw_base >= 0 && iw_base < input_width) {
                const int idx = input_batch_offset + input_channel_offset + (ih_base + dilation) * input_width + iw_base;
                max_val = std::max(max_val, input[idx]);
            }
            if (ih_base + dilation >= 0 && ih_base + dilation < input_height && iw_base + dilation >= 0 && iw_base + dilation < input_width) {
                const int idx = input_batch_offset + input_channel_offset + (ih_base + dilation) * input_width + (iw_base + dilation);
                max_val = std::max(max_val, input[idx]);
            }
        }
        else if constexpr (KERNEL_SIZE == 3) {
            const int ih_base = oh * stride - padding;
            const int iw_base = ow * stride - padding;

            for (int i = 0; i < 3; i++) {
                const int ih = ih_base + i * dilation;
                if (ih >= 0 && ih < input_height) {
                    const int ih_offset = ih * input_width;
                    for (int j = 0; j < 3; j++) {
                        const int iw = iw_base + j * dilation;
                        if (iw >= 0 && iw < input_width) {
                            const int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                            max_val = std::max(max_val, input[idx]);
                        }
                    }
                }
            }
        }
        else {
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                const int ih = oh * stride - padding + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    const int ih_offset = ih * input_width;
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        const int iw = ow * stride - padding + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            const int idx = input_batch_offset + input_channel_offset + ih_offset + iw;
                            max_val = std::max(max_val, input[idx]);
                        }
                    }
                }
            }
        }

        output[output_idx] = max_val;
    }
}

torch::Tensor max_pool2d_cpu_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cpu_forward", ([&] {
        if (kernel_size == 2) {
            max_pool2d_tuned_kernel_cpu<scalar_t, 2>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                batch_size, channels, input_height, input_width,
                output_height, output_width, stride, padding, dilation);
        }
        else if (kernel_size == 3) {
            max_pool2d_tuned_kernel_cpu<scalar_t, 3>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                batch_size, channels, input_height, input_width,
                output_height, output_width, stride, padding, dilation);
        }
        else {
            max_pool2d_tuned_kernel_cpu<scalar_t, -1>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                batch_size, channels, input_height, input_width,
                output_height, output_width, stride, padding, dilation);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cpu_forward, "Max Pool 2D forward with tuned block size (CPU)");
}
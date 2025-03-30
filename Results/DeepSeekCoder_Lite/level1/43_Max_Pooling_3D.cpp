#include <torch/extension.h>
#include <vector>
#include <limits>
#include <cmath>
#include <omp.h>

template <typename scalar_t, int KERNEL_SIZE>
void maxpool3d_unrolled_kernel_cpu(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int stride,
    const int padding,
    const int dilation) {

    const int output_size = batch_size * channels * output_d * output_h * output_w;
    #pragma omp parallel for
    for (int i = 0; i < output_size; i++) {
        int b = i / (channels * output_d * output_h * output_w);
        int c = (i / (output_d * output_h * output_w)) % channels;
        int d_out = (i / (output_h * output_w)) % output_d;
        int h_out = (i / output_w) % output_h;
        int w_out = i % output_w;

        const int d_start = d_out * stride - padding;
        const int h_start = h_out * stride - padding;
        const int w_start = w_out * stride - padding;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        int max_index = -1;

        if constexpr (KERNEL_SIZE <= 4) {
            #define UNROLL_KERNEL(kd, kh, kw) \
            { \
                const int d_in = d_start + kd * dilation; \
                const int h_in = h_start + kh * dilation; \
                const int w_in = w_start + kw * dilation; \
                if (d_in >= 0 && d_in < input_d && h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) { \
                    const int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in; \
                    const scalar_t val = input[input_idx]; \
                    if (val > max_val) { \
                        max_val = val; \
                        max_index = input_idx; \
                    } \
                } \
            }

            #pragma omp parallel for
            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        UNROLL_KERNEL(kd, kh, kw)
                    }
                }
            }
            #undef UNROLL_KERNEL
        } else {
            #pragma omp parallel for
            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                const int d_in = d_start + kd * dilation;
                if (d_in >= 0 && d_in < input_d) {
                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                        const int h_in = h_start + kh * dilation;
                        if (h_in >= 0 && h_in < input_h) {
                            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                                const int w_in = w_start + kw * dilation;
                                if (w_in >= 0 && w_in < input_w) {
                                    const int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                                    const scalar_t val = input[input_idx];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_index = input_idx;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        const int output_idx = (((b * channels + c) * output_d + d_out) * output_h + h_out) * output_w + w_out;
        output[output_idx] = max_val;
        if (indices != nullptr) {
            indices[output_idx] = max_index;
        }
    }
}

torch::Tensor max_pool3d_cpu_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {

    auto input_sizes = input.sizes();
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_d = input_sizes[2];
    const int input_h = input_sizes[3];
    const int input_w = input_sizes[4];

    const int output_d = ceil_mode ?
        ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_h = ceil_mode ?
        ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_w = ceil_mode ?
        ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch.kLong)) :
        torch::Tensor();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cpu", ([&] {
        switch(kernel_size) {
            case 2:
                maxpool3d_unrolled_kernel_cpu<scalar_t, 2>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    return_indices ? indices.data_ptr<int64_t>() : nullptr,
                    batch_size, channels, input_d, input_h, input_w,
                    output_d, output_h, output_w, stride, padding, dilation);
                break;
            case 3:
                maxpool3d_unrolled_kernel_cpu<scalar_t, 3>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    return_indices ? indices.data_ptr<int64_t>() : nullptr,
                    batch_size, channels, input_d, input_h, input_w,
                    output_d, output_h, output_w, stride, padding, dilation);
                break;
            case 4:
                maxpool3d_unrolled_kernel_cpu<scalar_t, 4>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    return_indices ? indices.data_ptr<int64_t>() : nullptr,
                    batch_size, channels, input_d, input_h, input_w,
                    output_d, output_h, output_w, stride, padding, dilation);
                break;
            default:
                maxpool3d_unrolled_kernel_cpu<scalar_t, 8>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    return_indices ? indices.data_ptr<int64_t>() : nullptr,
                    batch_size, channels, input_d, input_h, input_w,
                    output_d, output_h, output_w, stride, padding, dilation);
        }
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cpu_forward, "Max Pool 3D forward with unrolled loops (CPU)");
}
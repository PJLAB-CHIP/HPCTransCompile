#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Host forward function setting up convolution parameters and launching kernels
at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(weight.is_cpu(), "Weight must be a CPU tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cpu(), "Bias must be a CPU tensor");
    }

    // Input dimensions: [batch, in_channels, in_d, in_h, in_w]
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    // Weight dimensions: [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Calculate output dimensions using standard convolution formula
    int out_d = (in_d + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_d, out_h, out_w}, options);

    int total_elements = batch_size * out_channels * out_d * out_h * out_w;

    int in_channels_per_group = in_channels / groups;

    // Multi-threaded loop for convolution
    #pragma omp parallel for
    for (int idx = 0; idx < total_elements; ++idx) {
        // Decompose linear index into 5D indices: [b, oc, od, oh, ow]
        int w_idx = idx % out_w;
        int tmp = idx / out_w;
        int h_idx = tmp % out_h;
        tmp /= out_h;
        int d_idx = tmp % out_d;
        tmp /= out_d;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        // Determine the group and the corresponding input channels
        int group = oc / (out_channels / groups);
        int in_channel_base = group * in_channels_per_group;

        // Compute the top-left-front corner of the receptive field in the input
        int in_d_base = d_idx * stride - padding;
        int in_h_base = h_idx * stride - padding;
        int in_w_base = w_idx * stride - padding;

        // Precompute valid kernel loop bounds for depth
        int kd_start = 0;
        if (in_d_base < 0) {
            kd_start = (-in_d_base + dilation - 1) / dilation;
        }
        int kd_end = kernel_d;
        if (in_d_base + (kernel_d - 1) * dilation >= in_d) {
            kd_end = (in_d - in_d_base + dilation - 1) / dilation;
            if(kd_end > kernel_d) kd_end = kernel_d;
        }

        // Precompute valid kernel loop bounds for height
        int kh_start = 0;
        if (in_h_base < 0) {
            kh_start = (-in_h_base + dilation - 1) / dilation;
        }
        int kh_end = kernel_h;
        if (in_h_base + (kernel_h - 1) * dilation >= in_h) {
            kh_end = (in_h - in_h_base + dilation - 1) / dilation;
            if(kh_end > kernel_h) kh_end = kernel_h;
        }

        // Precompute valid kernel loop bounds for width
        int kw_start = 0;
        if (in_w_base < 0) {
            kw_start = (-in_w_base + dilation - 1) / dilation;
        }
        int kw_end = kernel_w;
        if (in_w_base + (kernel_w - 1) * dilation >= in_w) {
            kw_end = (in_w - in_w_base + dilation - 1) / dilation;
            if(kw_end > kernel_w) kw_end = kernel_w;
        }

        // Perform convolution sum
        scalar_t sum = 0;
        for (int ic = 0; ic < in_channels_per_group; ++ic) {
            int in_channel = in_channel_base + ic;
            for (int kd = kd_start; kd < kd_end; ++kd) {
                int id = in_d_base + kd * dilation;
                for (int kh = kh_start; kh < kh_end; ++kh) {
                    int ih = in_h_base + kh * dilation;
                    for (int kw = kw_start; kw < kw_end; ++kw) {
                        int iw = in_w_base + kw * dilation;
                        int input_idx = (((b * in_channels + in_channel) * in_d + id) * in_h + ih) * in_w + iw;
                        int weight_idx = (((oc * in_channels_per_group + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }

    if (bias.defined()) {
        // Multi-threaded loop for bias addition
        #pragma omp parallel for
        for (int idx = 0; idx < total_elements; ++idx) {
            // Remove the width dimension to extract the channel index
            int tmp = idx / out_w;
            int oc = tmp % out_channels;
            output[idx] += bias[oc];
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward CPU kernel");
}
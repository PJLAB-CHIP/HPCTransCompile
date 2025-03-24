#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ spatial_bias,
    float scaling_factor,
    int stride,
    int padding,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int out_depth,
    int out_height,
    int out_width,
    float* __restrict__ output
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_depth * out_height * out_width) return;

    const int w = idx % out_width;
    const int h = (idx / out_width) % out_height;
    const int d = (idx / (out_width * out_height)) % out_depth;
    const int b = idx / (out_width * out_height * out_depth);

    float total = 0.0f;

    for (int oc = 0; oc < out_channels; ++oc) {
        float channel_val = conv_bias[oc];
        
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int in_d_unclamped = (d - kd + padding) / stride;
                    const int in_h_unclamped = (h - kh + padding) / stride;
                    const int in_w_unclamped = (w - kw + padding) / stride;

                    const bool stride_valid = 
                        ((d - kd + padding) % stride == 0) &&
                        ((h - kh + padding) % stride == 0) &&
                        ((w - kw + padding) % stride == 0);

                    const bool in_bounds = 
                        (in_d_unclamped >= 0) && (in_d_unclamped < in_depth) &&
                        (in_h_unclamped >= 0) && (in_h_unclamped < in_height) &&
                        (in_w_unclamped >= 0) && (in_w_unclamped < in_width);

                    const float valid = (stride_valid && in_bounds) ? 1.0f : 0.0f;

                    const int in_d = max(0, min(in_depth - 1, in_d_unclamped));
                    const int in_h = max(0, min(in_height - 1, in_h_unclamped));
                    const int in_w = max(0, min(in_width - 1, in_w_unclamped));

                    for (int ic = 0; ic < in_channels; ++ic) {
                        const int input_idx = (((b * in_channels + ic) * in_depth + in_d)
                                            * in_height + in_h) * in_width + in_w;
                        const int weight_idx = (((ic * out_channels + oc) * kernel_d + kd)
                                            * kernel_h + kh) * kernel_w + kw;
                        
                        channel_val += input[input_idx] * conv_weight[weight_idx] * valid;
                    }
                }
            }
        }
        total += channel_val;
    }

    const float mean_val = total / out_channels;
    const int spatial_idx = d * out_height * out_width + h * out_width + w;
    const float biased = mean_val + spatial_bias[spatial_idx];
    output[idx] = tanhf(1.0f) * scaling_factor;
}

torch::Tensor forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& spatial_bias,
    float scaling_factor,
    int stride,
    int padding
) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor");
    TORCH_CHECK(conv_weight.dim() == 5, "Conv weight must be 5D tensor");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_channels = conv_weight.size(1);
    const int kernel_d = conv_weight.size(2);
    const int kernel_h = conv_weight.size(3);
    const int kernel_w = conv_weight.size(4);

    const int out_depth = (in_depth - 1) * stride + kernel_d - 2 * padding;
    const int out_height = (in_height - 1) * stride + kernel_h - 2 * padding;
    const int out_width = (in_width - 1) * stride + kernel_w - 2 * padding;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    torch::Tensor output = torch::empty({batch_size, 1, out_depth, out_height, out_width}, options);

    const int threads = 512;
    const int total_elements = batch_size * out_depth * out_height * out_width;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_operations_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        spatial_bias.data_ptr<float>(),
        scaling_factor,
        stride,
        padding,
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        kernel_d,
        kernel_h,
        kernel_w,
        out_depth,
        out_height,
        out_width,
        output.data_ptr<float>()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Fused Transposed Conv3D Operations (CUDA)");
}
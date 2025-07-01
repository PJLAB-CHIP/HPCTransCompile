#include <torch/extension.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D_in,
    int H_in,
    int W_in,
    int D_out,
    int H_out,
    int W_out,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * D_out * H_out * W_out;

    if (idx < total_elements) {
        // Compute output indices from global index
        int n = idx / (out_channels * D_out * H_out * W_out);
        int c_out = (idx / (D_out * H_out * W_out)) % out_channels;
        int d_out = (idx / (H_out * W_out)) % D_out;
        int h_out = (idx / W_out) % H_out;
        int w_out = idx % W_out;

        // Determine group and channel offsets
        int g = c_out / (out_channels / groups);
        int c_in_start = g * (in_channels / groups);
        int c_in_end = c_in_start + (in_channels / groups);
        int c_out_g = c_out % (out_channels / groups);

        float sum = 0.0f;

        // Loop over filter dimensions
        for (int k_d = 0; k_d < kernel_size; ++k_d) {
            int d_temp = d_out + padding - k_d * dilation;
            if (d_temp >= 0 && d_temp % stride == 0) {
                int d_in = d_temp / stride;
                if (d_in >= 0 && d_in < D_in) {
                    for (int k_h = 0; k_h < kernel_size; ++k_h) {
                        int h_temp = h_out + padding - k_h * dilation;
                        if (h_temp >= 0 && h_temp % stride == 0) {
                            int h_in = h_temp / stride;
                            if (h_in >= 0 && h_in < H_in) {
                                for (int k_w = 0; k_w < kernel_size; ++k_w) {
                                    int w_temp = w_out + padding - k_w * dilation;
                                    if (w_temp >= 0 && w_temp % stride == 0) {
                                        int w_in = w_temp / stride;
                                        if (w_in >= 0 && w_in < W_in) {
                                            // Loop over input channels in the group
                                            for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
                                                int input_idx = n * (in_channels * D_in * H_in * W_in) +
                                                               c_in * (D_in * H_in * W_in) +
                                                               d_in * (H_in * W_in) +
                                                               h_in * W_in +
                                                               w_in;
                                                int weight_idx = c_in * ((out_channels / groups) * kernel_size * kernel_size * kernel_size) +
                                                                c_out_g * (kernel_size * kernel_size * kernel_size) +
                                                                k_d * (kernel_size * kernel_size) +
                                                                k_h * kernel_size +
                                                                k_w;
                                                sum += input[input_idx] * weight[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[c_out];
        }

        // Write result to output
        int output_idx = n * (out_channels * D_out * H_out * W_out) +
                        c_out * (D_out * H_out * W_out) +
                        d_out * (H_out * W_out) +
                        h_out * W_out +
                        w_out;
        output[output_idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups
) {
    // Extract input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    // Compute output channels and kernel size from weight
    int out_channels = weight.size(1) * groups;  // weight shape: (in_channels, out_channels // groups, K, K, K)
    int kernel_size = weight.size(2);  // Assuming kernel_size is same for all dimensions

    // Compute output dimensions
    int D_out = (D_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, D_out, H_out, W_out}, input.options());

    // Get pointers to tensor data
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Launch kernel
    int total = batch_size * out_channels * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        out_channels,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        kernel_size,
        stride,
        padding,
        dilation,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3D forward");
}
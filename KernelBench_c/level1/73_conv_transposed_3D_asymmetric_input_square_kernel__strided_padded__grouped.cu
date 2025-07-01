#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

/**
 * CUDA kernel for 3D transposed convolution.
 * Each thread computes one output element by summing contributions from the input and weights.
 */
__global__ void conv_transpose3d_kernel(
    const float* x,                // Input tensor: (batch_size, in_channels, D_in, H_in, W_in)
    const float* weight,           // Weight tensor: (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
    const float* bias,             // Bias tensor: (out_channels) or nullptr if no bias
    float* y,                      // Output tensor: (batch_size, out_channels, D_out, H_out, W_out)
    int batch_size,                // Number of samples in the batch
    int in_channels,               // Number of input channels
    int out_channels,              // Number of output channels
    int groups,                    // Number of groups
    int D_in, int H_in, int W_in,  // Input spatial dimensions
    int kernel_size,               // Kernel size (assumed square)
    int stride,                    // Stride (same for all dimensions)
    int padding,                   // Padding (same for all dimensions)
    int D_out, int H_out, int W_out// Output spatial dimensions
) {
    // Compute global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * D_out * H_out * W_out;

    if (index < total_elements) {
        // Extract output indices from the global index
        int w_out = index % W_out;
        index /= W_out;
        int h_out = index % H_out;
        index /= H_out;
        int d_out = index % D_out;
        index /= D_out;
        int c_out = index % out_channels;
        int b = index / out_channels;

        // Initialize output value with bias if provided, otherwise zero
        float value = (bias != nullptr) ? bias[c_out] : 0.0f;

        // Compute group parameters
        int G = groups;
        int in_channels_per_group = in_channels / G;
        int out_channels_per_group = out_channels / G;
        int g = c_out / out_channels_per_group;         // Group index
        int c_out_sub = c_out % out_channels_per_group; // Sub-index within the group's output channels

        // Loop over input channels within the group
        for (int c_in_local = 0; c_in_local < in_channels_per_group; c_in_local++) {
            int c_in = g * in_channels_per_group + c_in_local;

            // Loop over kernel dimensions
            for (int kd = 0; kd < kernel_size; kd++) {
                int d_temp = d_out - kd + padding;
                if (d_temp >= 0 && d_temp % stride == 0) {
                    int d_in = d_temp / stride;
                    if (d_in < D_in) {  // d_in >= 0 since d_temp >= 0
                        for (int kh = 0; kh < kernel_size; kh++) {
                            int h_temp = h_out - kh + padding;
                            if (h_temp >= 0 && h_temp % stride == 0) {
                                int h_in = h_temp / stride;
                                if (h_in < H_in) {
                                    for (int kw = 0; kw < kernel_size; kw++) {
                                        int w_temp = w_out - kw + padding;
                                        if (w_temp >= 0 && w_temp % stride == 0) {
                                            int w_in = w_temp / stride;
                                            if (w_in < W_in) {
                                                // Compute input and weight indices
                                                int x_idx = b * (in_channels * D_in * H_in * W_in) +
                                                            c_in * (D_in * H_in * W_in) +
                                                            d_in * (H_in * W_in) +
                                                            h_in * W_in +
                                                            w_in;
                                                int weight_idx = c_in * (out_channels_per_group * kernel_size * kernel_size * kernel_size) +
                                                                 c_out_sub * (kernel_size * kernel_size * kernel_size) +
                                                                 kd * (kernel_size * kernel_size) +
                                                                 kh * kernel_size +
                                                                 kw;
                                                // Accumulate contribution
                                                value += x[x_idx] * weight[weight_idx];
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

        // Write result to output tensor
        int y_idx = b * (out_channels * D_out * H_out * W_out) +
                    c_out * (D_out * H_out * W_out) +
                    d_out * (H_out * W_out) +
                    h_out * W_out +
                    w_out;
        y[y_idx] = value;
    }
}

/**
 * C++ wrapper to launch the CUDA kernel.
 * Computes output shape and allocates the output tensor.
 */
torch::Tensor conv_transpose3d_cuda(
    const torch::Tensor& x,          // Input tensor
    const torch::Tensor& weight,     // Weight tensor
    const torch::Tensor& bias,       // Bias tensor (optional)
    int stride,                      // Stride
    int padding,                     // Padding
    int output_padding,              // Output padding
    int groups                       // Number of groups
) {
    // Extract input dimensions
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto D_in = x.size(2);
    auto H_in = x.size(3);
    auto W_in = x.size(4);
    auto out_channels = weight.size(1) * groups;  // out_channels = (out_channels // groups) * groups
    auto kernel_size = weight.size(2);

    // Compute output dimensions (dilation=1 as per PyTorch code)
    int D_out = (D_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Allocate output tensor
    auto y = torch::empty({batch_size, out_channels, D_out, H_out, W_out}, x.options());

    // Get pointers to tensor data
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* y_ptr = y.data_ptr<float>();

    // Launch kernel
    int total_elements = batch_size * out_channels * D_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    conv_transpose3d_kernel<<<grid_size, block_size>>>(
        x_ptr, weight_ptr, bias_ptr, y_ptr,
        batch_size, in_channels, out_channels, groups,
        D_in, H_in, W_in,
        kernel_size,
        stride,
        padding,
        D_out, H_out, W_out
    );

    // Check for CUDA errors (optional, for debugging)
    cudaDeviceSynchronize();
    return y;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose3d_cuda, "3D transposed convolution CUDA implementation");
}
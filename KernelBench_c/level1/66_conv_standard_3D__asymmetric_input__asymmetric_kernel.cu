#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for 3D convolution
template <typename scalar_t>
__global__ void conv3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int groups) {

    // Global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_depth = (in_depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    const int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    const int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx >= total_threads) return;

    // Compute output coordinates
    const int w = idx % out_width;
    const int h = (idx / out_width) % out_height;
    const int d = (idx / (out_width * out_height)) % out_depth;
    const int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
    const int n = idx / (out_width * out_height * out_depth * out_channels);

    // Group handling
    const int in_channels_per_group = in_channels / groups;
    const int out_channels_per_group = out_channels / groups;
    const int group = c_out / out_channels_per_group;
    const int c_out_local = c_out % out_channels_per_group;

    scalar_t value = bias ? bias[c_out] : scalar_t(0);

    // Convolution computation
    for (int kd = 0; kd < kernel_d; kd++) {
        int z = d * stride_d - padding_d + kd * dilation_d;
        if (z < 0 || z >= in_depth) continue;

        for (int kh = 0; kh < kernel_h; kh++) {
            int y = h * stride_h - padding_h + kh * dilation_h;
            if (y < 0 || y >= in_height) continue;

            for (int kw = 0; kw < kernel_w; kw++) {
                int x = w * stride_w - padding_w + kw * dilation_w;
                if (x < 0 || x >= in_width) continue;

                for (int c_in_local = 0; c_in_local < in_channels_per_group; c_in_local++) {
                    int c_in = group * in_channels_per_group + c_in_local;
                    scalar_t val = input[n * in_channels * in_depth * in_height * in_width +
                                         c_in * in_depth * in_height * in_width +
                                         z * in_height * in_width +
                                         y * in_width + x];
                    scalar_t wgt = weight[c_out * in_channels_per_group * kernel_d * kernel_h * kernel_w +
                                          c_in_local * kernel_d * kernel_h * kernel_w +
                                          kd * kernel_h * kernel_w +
                                          kh * kernel_w + kw];
                    value += val * wgt;
                }
            }
        }
    }
    output[idx] = value;
}

// Helper function to launch the kernel
torch::Tensor conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups) {

    // Ensure tensors are on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");

    // Input dimensions: [batch_size, in_channels, in_depth, in_height, in_width]
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    // Weight dimensions: [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    // Compute output dimensions
    const int out_depth = (in_depth + 2 * padding[0] - dilation[0] * (kernel_d - 1) - 1) / stride[0] + 1;
    const int out_height = (in_height + 2 * padding[1] - dilation[1] * (kernel_h - 1) - 1) / stride[1] + 1;
    const int out_width = (in_width + 2 * padding[2] - dilation[2] * (kernel_w - 1) - 1) / stride[2] + 1;

    // Output tensor
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Launch parameters
    const int threads = 256;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int blocks = (total_elements + threads - 1) / threads;
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for transposed 3D convolution
template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int groups) {

    // Global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_depth = in_depth * stride_d - 2 * padding_d + kernel_d - 1 + output_padding_d;
    const int out_height = in_height * stride_h - 2 * padding_h + kernel_h - 1 + output_padding_h;
    const int out_width = in_width * stride_w - 2 * padding_w + kernel_w - 1 + output_padding_w;

    const int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx >= total_threads) return;

    // Compute output coordinates
    const int w_out = idx % out_width;
    const int h_out = (idx / out_width) % out_height;
    const int d_out = (idx / (out_width * out_height)) % out_depth;
    const int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
    const int n = idx / (out_width * out_height * out_depth * out_channels);

    // Group handling
    const int in_channels_per_group = in_channels / groups;
    const int out_channels_per_group = out_channels / groups;
    const int group = c_out / out_channels_per_group;
    const int c_out_local = c_out % out_channels_per_group;

    scalar_t value = bias ? bias[c_out] : scalar_t(0);

    // Transposed convolution computation
    for (int kd = 0; kd < kernel_d; kd++) {
        int d_in = (d_out + padding_d - kd) / stride_d;
        if ((d_out + padding_d - kd) % stride_d != 0 || d_in < 0 || d_in >= in_depth) continue;

        for (int kh = 0; kh < kernel_h; kh++) {
            int h_in = (h_out + padding_h - kh) / stride_h;
            if ((h_out + padding_h - kh) % stride_h != 0 || h_in < 0 || h_in >= in_height) continue;

            for (int kw = 0; kw < kernel_w; kw++) {
                int w_in = (w_out + padding_w - kw) / stride_w;
                if ((w_out + padding_w - kw) % stride_w != 0 || w_in < 0 || w_in >= in_width) continue;

                for (int c_in_local = 0; c_in_local < in_channels_per_group; c_in_local++) {
                    int c_in = group * in_channels_per_group + c_in_local;
                    scalar_t val = input[n * in_channels * in_depth * in_height * in_width +
                                         c_in * in_depth * in_height * in_width +
                                         d_in * in_height * in_width +
                                         h_in * in_width + w_in];
                    scalar_t wgt = weight[c_in * out_channels_per_group * kernel_d * kernel_h * kernel_w +
                                          c_out_local * kernel_d * kernel_h * kernel_w +
                                          kd * kernel_h * kernel_w +
                                          kh * kernel_w + kw];
                    value += val * wgt;
                }
            }
        }
    }
    output[idx] = value;
}

// Helper function to launch the kernel
torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    int64_t groups) {

    // Ensure tensors are on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");

    // Input dimensions: [batch_size, in_channels, in_depth, in_height, in_width]
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    // Weight dimensions: [in_channels, out_channels/groups, kernel_d, kernel_h, kernel_w]
    const int out_channels = weight.size(1) * groups;
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    // Compute output dimensions
    const int out_depth = in_depth * stride[0] - 2 * padding[0] + kernel_d - 1 + output_padding[0];
    const int out_height = in_height * stride[1] - 2 * padding[1] + kernel_h - 1 + output_padding[1];
    const int out_width = in_width * stride[2] - 2 * padding[2] + kernel_w - 1 + output_padding[2];

    // Output tensor
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Launch parameters
    const int threads = 256;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int blocks = (total_elements + threads - 1) / threads;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_forward_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_depth,
            in_height,
            in_width,
            kernel_d,
            kernel_h,
            kernel_w,
            stride[0],
            stride[1],
            stride[2],
            padding[0],
            padding[1],
            padding[2],
            output_padding[0],
            output_padding[1],
            output_padding[2],
            groups);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}

// PyBind11 module definition
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
    return conv_transpose3d_forward_cuda(input, weight, bias, stride, padding, output_padding, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 3D Convolution Forward (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"), py::arg("groups"));
}
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward_cuda", ([&] {
        conv3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_depth,
            in_height,
            in_width,
            kernel_d,
            kernel_h,
            kernel_w,
            stride[0],
            stride[1],
            stride[2],
            padding[0],
            padding[1],
            padding[2],
            dilation[0],
            dilation[1],
            dilation[2],
            groups);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}

// PyBind11 module definition
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
    return conv3d_forward_cuda(input, weight, bias, stride, padding, dilation, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Convolution Forward (CUDA)");
}
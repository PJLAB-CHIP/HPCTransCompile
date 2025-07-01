#include <torch/extension.h>
#include <pybind11/pybind11.h>

__global__ void conv2d_optimized_kernel(const float* input, const float* weight,
                                        const float* bias, float* output,
                                        int batch_size, int in_channels, int out_channels,
                                        int height, int width, int kernel_size,
                                        int stride, int padding) {
    const int n = blockIdx.x;
    const int oc = blockIdx.y;
    const int oh = blockIdx.z * blockDim.y + threadIdx.y;
    const int ow = blockIdx.z * blockDim.x + threadIdx.x;

    const int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (width + 2 * padding - kernel_size) / stride + 1;

    if (oh < output_height && ow < output_width) {
        float sum = bias[oc];

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = oh * stride - padding + kh;
                    int w_in = ow * stride - padding + kw;

                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        float input_val = input[((n * in_channels + ic) * height + h_in) * width + w_in];
                        float weight_val = weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
                        sum += input_val * weight_val;
                    }
                }
            }
        }

        output[((n * out_channels + oc) * output_height + oh) * output_width + ow] = sum;
    }
}

// Wrapper function for the optimized conv2d kernel
torch::Tensor optimized_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                    int stride, int padding) {
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(32, 8);  // 32 threads for width, 8 threads for height
    dim3 blocks(batch_size, out_channels, (output_height + threads.y - 1) / threads.y);

    conv2d_optimized_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(),
                                                 bias.data_ptr<float>(), output.data_ptr<float>(),
                                                 batch_size, in_channels, out_channels,
                                                 height, width, kernel_size, stride, padding);
    cudaDeviceSynchronize();

    return output;
}

torch::Tensor module_fn(torch::Tensor x, py::object params, bool is_training) {
    auto get_param = [&](const std::string& key) {
        return params.attr("__getitem__")(key.c_str()).cast<torch::Tensor>();
    };

    auto conv1_w = get_param("conv1_w");
    auto conv1_b = get_param("conv1_b");
    x = optimized_conv2d_cuda(x, conv1_w, conv1_b, /*stride=*/2, /*padding=*/3);
    x = at::relu(x);
    x = at::max_pool2d(x, /*kernel_size=*/3, /*stride=*/2, /*padding=*/1);

    auto conv2_w = get_param("conv2_w");
    auto conv2_b = get_param("conv2_b");
    x = at::conv2d(x, conv2_w, conv2_b);
    x = at::relu(x);

    auto conv3_w = get_param("conv3_w");
    auto conv3_b = get_param("conv3_b");
    x = at::conv2d(x, conv3_w, conv3_b, /*stride=*/1, /*padding=*/1);
    x = at::relu(x);
    x = at::max_pool2d(x, /*kernel_size=*/3, /*stride=*/2, /*padding=*/1);

    auto inception_module_fn = [&](torch::Tensor input, const std::string& prefix) {
        auto conv1x1_w = get_param(prefix + "_1x1_w");
        auto conv1x1_b = get_param(prefix + "_1x1_b");
        auto branch1x1 = at::conv2d(input, conv1x1_w, conv1x1_b);

        auto conv3x3_reduce_w = get_param(prefix + "_3x3_reduce_w");
        auto conv3x3_reduce_b = get_param(prefix + "_3x3_reduce_b");
        auto conv3x3_w = get_param(prefix + "_3x3_w");
        auto conv3x3_b = get_param(prefix + "_3x3_b");
        auto branch3x3 = at::conv2d(input, conv3x3_reduce_w, conv3x3_reduce_b);
        branch3x3 = at::conv2d(branch3x3, conv3x3_w, conv3x3_b, /*stride=*/1, /*padding=*/1);

        auto conv5x5_reduce_w = get_param(prefix + "_5x5_reduce_w");
        auto conv5x5_reduce_b = get_param(prefix + "_5x5_reduce_b");
        auto conv5x5_w = get_param(prefix + "_5x5_w");
        auto conv5x5_b = get_param(prefix + "_5x5_b");
        auto branch5x5 = at::conv2d(input, conv5x5_reduce_w, conv5x5_reduce_b);
        branch5x5 = at::conv2d(branch5x5, conv5x5_w, conv5x5_b, /*stride=*/1, /*padding=*/2);

        auto pool_proj_w = get_param(prefix + "_pool_proj_w");
        auto pool_proj_b = get_param(prefix + "_pool_proj_b");
        auto branch_pool = at::max_pool2d(input, /*kernel_size=*/3, /*stride=*/1, /*padding=*/1);
        branch_pool = at::conv2d(branch_pool, pool_proj_w, pool_proj_b);

        return at::cat({branch1x1, branch3x3, branch5x5, branch_pool}, 1);
    };

    x = inception_module_fn(x, "3a");
    x = inception_module_fn(x, "3b");
    x = at::max_pool2d(x, /*kernel_size=*/3, /*stride=*/2, /*padding=*/1);

    x = inception_module_fn(x, "4a");
    x = inception_module_fn(x, "4b");
    x = inception_module_fn(x, "4c");
    x = inception_module_fn(x, "4d");
    x = inception_module_fn(x, "4e");
    x = at::max_pool2d(x, /*kernel_size=*/3, /*stride=*/2, /*padding=*/1);

    x = inception_module_fn(x, "5a");
    x = inception_module_fn(x, "5b");
    
    x = at::adaptive_avg_pool2d(x, {1, 1});
    x = x.view({x.size(0), -1});
    x = at::dropout(x, /*p=*/0.0, /*train=*/is_training);

    auto fc_w = get_param("fc_w");
    auto fc_b = get_param("fc_b");
    x = at::linear(x, fc_w, fc_b);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Module forward function",
          py::arg("x"), py::arg("params"), py::arg("is_training"));
}

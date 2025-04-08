#include <torch/extension.h>
#include <map>
#include <string>
#include <vector>
#include <cmath>

using namespace torch;

// Function to perform reduction sum across a vector
template<typename T>
T reduce_sum(const std::vector<T>& vec) {
    T sum = 0;
    for (const auto& val : vec) {
        sum += val;
    }
    return sum;
}

// Function to perform batch normalization on a tensor
void batch_norm_cpu(Tensor& out, const Tensor& in, const Tensor& weight, const Tensor& bias,
                    const Tensor& mean, const Tensor& var, bool is_training) {
    auto in_data = in.data_ptr<float>();
    auto out_data = out.data_ptr<float>();
    auto weight_data = weight.data_ptr<float>();
    auto bias_data = bias.data_ptr<float>();
    auto mean_data = mean.data_ptr<float>();
    auto var_data = var.data_ptr<float>();

    int64_t size = in.numel();
    for (int64_t i = 0; i < size; ++i) {
        float normalized = (in_data[i] - mean_data[i]) * std::sqrt(1.0f / (var_data[i] + 1e-5f));
        out_data[i] = normalized * weight_data[i] + bias_data[i];
    }
}

// Function to perform depthwise convolution
Tensor conv2d_cpu(const Tensor& input, const Tensor& weight, int stride, int padding, int groups) {
    // Implementing a simple 2D convolution without using any optimized libraries
    auto input_data = input.data_ptr<float>();
    auto weight_data = weight.data_ptr<float>();
    auto output = torch::zeros_like(input);
    auto output_data = output.data_ptr<float>();

    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t out_channels = weight.size(0);
    int64_t kernel_size = weight.size(2);
    int64_t height = input.size(2);
    int64_t width = input.size(3);

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t g = 0; g < groups; ++g) {
            for (int64_t oc = g; oc < out_channels; oc += groups; {
                for (int64_t ic = g; ic < in_channels; ic += groups; {
                    for (int64_t h = 0; h < height; ++h) {
                        for (int64_t w = 0; w < width; ++w) {
                            int64_t oh = h - kernel_size / 2 + padding;
                            int64_t ow = w - kernel_size / 2 + padding;
                            if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                                output_data[b * out_channels * height * width + oc * height * width + oh * width + ow] +=
                                    input_data[b * in_channels * height * width + ic * height * width + h * width + w] *
                                    weight_data[oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + h * kernel_size + w];
                            }
                        }
                    }
                }
            }
        }
    }
    return output;
}

// Function to perform the MBConv block
Tensor mbconv_block_cpu(Tensor x, std::map<std::string, Tensor>& params, int stride, int expand_ratio, bool is_training) {
    int64_t in_channels = x.size(1);
    int64_t expanded_channels = in_channels * expand_ratio;

    if (expand_ratio != 1) {
        auto expand_conv_weight = params["expand_conv_weight"];
        x = conv2d_cpu(x, expand_conv_weight, 1, 0, 1);
        batch_norm_cpu(x, x, params["expand_bn_weight"], params["expand_bn_bias"],
                       params["expand_bn_mean"], params["expand_bn_var"], is_training);
        x = relu(x);
    }

    auto dw_conv_weight = params["dw_conv_weight"];
    x = conv2d_cpu(x, dw_conv_weight, stride, 1, expanded_channels);
    batch_norm_cpu(x, x, params["dw_bn_weight"], params["dw_bn_bias"],
                   params["dw_bn_mean"], params["dw_bn_var"], is_training);
    x = relu(x);

    auto se = adaptive_avg_pool2d(x, {1, 1});
    se = conv2d_cpu(se, params["se_reduce_weight"], 1, 0, 1);
    se = relu(se);
    se = conv2d_cpu(se, params["se_expand_weight"], 1, 0, 1);
    se = sigmoid(se);
    x = x * se;

    auto project_conv_weight = params["project_conv_weight"];
    x = conv2d_cpu(x, project_conv_weight, 1, 0, 1);
    batch_norm_cpu(x, x, params["project_bn_weight"], params["project_bn_bias"],
                   params["project_bn_mean"], params["project_bn_var"], is_training);

    return x;
}

// Forward function for the model
Tensor forward_cpu(Tensor x, std::map<std::string, Tensor> params, bool is_training) {
    x = conv2d_cpu(x, params["conv1_weight"], 2, 1, 1);
    batch_norm_cpu(x, x, params["bn1_weight"], params["bn1_bias"],
                   params["bn1_mean"], params["bn1_var"], is_training);
    x = relu(x);

    const std::vector<std::pair<int, int>> mbconv_configs = {{1,3}, {2,6}, {2,6}, {2,6}, {1,6}};

    for (int i = 0; i < mbconv_configs.size(); ++i) {
        int block_num = i + 1;
        auto [stride, expand_ratio] = mbconv_configs[i];

        std::map<std::string, Tensor> block_params;
        std::string prefix = "mbconv" + std::to_string(block_num) + "_";

        for (const auto& pair : params) {
            if (pair.first.rfind(prefix, 0) == 0) {
                std::string key = pair.first.substr(prefix.length());
                block_params[key] = pair.second;
            }
        }

        x = mbconv_block_cpu(x, block_params, stride, expand_ratio, is_training);
    }

    x = conv2d_cpu(x, params["conv_final_weight"], 1, 0, 1);
    batch_norm_cpu(x, x, params["bn_final_weight"], params["bn_final_bias"],
                   params["bn_final_mean"], params["bn_final_var"], is_training);
    x = relu(x);
    x = adaptive_avg_pool2d(x, {1, 1});
    x = x.flatten(1);
    x = linear(x, params["fc_weight"], params["fc_bias"]);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "EfficientNetB2 forward with CPU implementation");
}

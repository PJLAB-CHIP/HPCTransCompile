#include <torch/extension.h>
#include <map>
#include <string>
#include <vector>
#include <omp.h>
#include <cmath>

using namespace torch;

template<int WARP_SIZE=32>
float warp_reduce_sum(float val, const std::vector<float>& data, int idx) {
    #pragma omp parallel for reduction(+:val)
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += data[idx + offset];
    }
    return val;
}

template<int WARP_SIZE=32>
void warp_batch_norm(float* out, const float* in,
                     const float* weight, const float* bias,
                     const float* mean, const float* var,
                     const int idx) {
    float normalized = (in[idx] - mean[idx]) * std::sqrt(1.0f / (var[idx] + 1e-5f));
    float result = normalized * weight[idx] + bias[idx];
    
    #pragma omp parallel for
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float temp = out[idx + offset];
        if (idx % (2 * offset) == 0) {
            out[idx] = temp;
        }
    }
    
    out[idx] = result;
}

Tensor mbconv_block(Tensor x, std::map<std::string, Tensor>& params, int stride, int expand_ratio, bool is_training) {
    int64_t in_channels = x.size(1);
    int64_t expanded_channels = in_channels * expand_ratio;

    if (expand_ratio != 1) {
        auto expand_conv_weight = params["expand_conv_weight"];
        x = conv2d(x, expand_conv_weight, Tensor(), 
                  {1}, at::IntArrayRef({0}), {1}, 1);
        x = batch_norm(
            x, params["expand_bn_weight"], params["expand_bn_bias"],
            params["expand_bn_mean"], params["expand_bn_var"],
            is_training, 0.1, 1e-5, true
        );
        x = relu(x);
    }

    auto dw_conv_weight = params["dw_conv_weight"];
    x = conv2d(x, dw_conv_weight, Tensor(), 
              {stride}, at::IntArrayRef({1}), {1}, expanded_channels);
    x = batch_norm(
        x, params["dw_bn_weight"], params["dw_bn_bias"],
        params["dw_bn_mean"], params["dw_bn_var"],
        is_training, 0.1, 1e-5, true
    );
    x = relu(x);

    auto se = adaptive_avg_pool2d(x, {1, 1});
    se = conv2d(se, params["se_reduce_weight"], Tensor(),
               {1}, at::IntArrayRef({0}));
    se = relu(se);
    se = conv2d(se, params["se_expand_weight"], Tensor(),
               {1}, at::IntArrayRef({0}));
    se = sigmoid(se);
    x = se;

    auto project_conv_weight = params["project_conv_weight"];
    x = conv2d(x, project_conv_weight, Tensor(),
              {1}, at::IntArrayRef({0}), {1}, 1);
    x = batch_norm(
        x, params["project_bn_weight"], params["project_bn_bias"],
        params["project_bn_mean"], params["project_bn_var"],
        is_training, 0.1, 1e-5, true
    );

    return x;
}

Tensor forward(Tensor x, std::map<std::string, Tensor> params, bool is_training) {
    x = conv2d(x, params["conv1_weight"], Tensor(),
              {2}, at::IntArrayRef({1}));
    x = batch_norm(
        x, params["bn1_weight"], params["bn1_bias"],
        params["bn1_mean"], params["bn1_var"],
        is_training, 0.1, 1e-5, true
    );
    x = relu(x);

    const std::vector<std::pair<int, int>> mbconv_configs = {{1,3}, {2,6}, {2,6}, {2,6}, {1,6}};
    
    #pragma omp parallel for
    for (int i = 0; i < mbconv_configs.size(); i++) {
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
        
        x = mbconv_block(x, block_params, stride, expand_ratio, is_training);
    }

    x = conv2d(x, params["conv_final_weight"], Tensor(),
              {1}, at::IntArrayRef({0}));
    x = batch_norm(
        x, params["bn_final_weight"], params["bn_final_bias"],
        params["bn_final_mean"], params["bn_final_var"],
        is_training, 0.1, 1e-5, true
    );
    x = relu(x);
    x = adaptive_avg_pool2d(x, {1, 1});
    x = x.flatten(1);
    x = linear(x, params["fc_weight"], params["fc_bias"]);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "EfficientNetB2 forward with OpenMP optimizations");
}
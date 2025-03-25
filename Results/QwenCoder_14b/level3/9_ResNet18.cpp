#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>

namespace py = pybind11;

// CPU version of the fused_add_relu_aligned_kernel
void fused_add_relu_aligned_cpu(
    const float* input,
    const float* identity,
    float* output,
    const int size
) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output[i] = fmaxf(input[i] + identity[i], 0.0f);
    }
}

void launch_fused_kernel(torch::Tensor& x, const torch::Tensor& identity) {
    const int size = x.numel();
    fused_add_relu_aligned_cpu(
        x.data_ptr<float>(),
        identity.data_ptr<float>(),
        x.data_ptr<float>(),
        size
    );
}

torch::Tensor basic_block_fn(
    torch::Tensor x,
    const torch::Tensor& conv1_w,
    const torch::Tensor& bn1_w,
    const torch::Tensor& bn1_b,
    const torch::Tensor& bn1_rm,
    const torch::Tensor& bn1_rv,
    const torch::Tensor& conv2_w,
    const torch::Tensor& bn2_w,
    const torch::Tensor& bn2_b,
    const torch::Tensor& bn2_rm,
    const torch::Tensor& bn2_rv,
    const torch::Tensor& downsample_conv_w,
    const torch::Tensor& downsample_bn_w,
    const torch::Tensor& downsample_bn_b,
    const torch::Tensor& downsample_bn_rm,
    const torch::Tensor& downsample_bn_rv,
    int64_t stride,
    bool is_training
) {
    torch::Tensor identity = x;

    x = torch::conv2d(x.contiguous(), conv1_w.contiguous(), 
                     /*bias=*/{}, /*stride=*/{stride, stride}, /*padding=*/{1, 1});

    x = torch::batch_norm(
        x,
        bn1_w,
        bn1_b,
        bn1_rm,
        bn1_rv,
        is_training,
        0.0,
        1e-5,
        true
    );

    x = torch::relu(x);

    x = torch::conv2d(x.contiguous(), conv2_w.contiguous(), 
                     /*bias=*/{}, /*stride=*/{1, 1}, /*padding=*/{1, 1});

    x = torch::batch_norm(
        x,
        bn2_w,
        bn2_b,
        bn2_rm,
        bn2_rv,
        is_training,
        0.0,
        1e-5,
        true
    );

    if (downsample_conv_w.defined()) {
        identity = torch::conv2d(identity.contiguous(), downsample_conv_w.contiguous(), 
                               /*bias=*/{}, /*stride=*/{stride, stride});
        identity = torch::batch_norm(
            identity,
            downsample_bn_w,
            downsample_bn_b,
            downsample_bn_rm,
            downsample_bn_rv,
            is_training,
            0.0,
            1e-5,
            true
        );
    }

    launch_fused_kernel(x, identity);
    return x;
}

torch::Tensor module_fn(torch::Tensor x, py::object params_py, bool is_training) {
    auto get_param = [&](const std::string& key) -> torch::Tensor {
        return params_py.attr("__getitem__")(key.c_str()).cast<torch::Tensor>().contiguous();
    };

    x = torch::conv2d(x.contiguous(), get_param("conv1_weight"), 
                     /*bias=*/{}, /*stride=*/{2, 2}, /*padding=*/{3, 3});

    x = torch::batch_norm(
        x,
        get_param("bn1_weight"),
        get_param("bn1_bias"),
        get_param("bn1_running_mean"),
        get_param("bn1_running_var"),
        is_training,
        0.0,
        1e-5,
        true
    );

    x = torch::relu(x);
    x = torch::max_pool2d(x, /*kernel_size=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1});

    for (int i = 1; i <= 4; ++i) {
        std::string layer_name = "layer" + std::to_string(i);
        for (int j = 0; j < 2; ++j) {
            std::string block_name = layer_name + "_" + std::to_string(j);
            int64_t stride = (i > 1 && j == 0) ? 2 : 1;

            std::string downsample_conv_key = block_name + "_downsample_0_weight";
            bool has_downsample = PyMapping_HasKeyString(params_py.ptr(), downsample_conv_key.c_str()) == 1;

            torch::Tensor downsample_conv_w, downsample_bn_w, downsample_bn_b, 
                         downsample_bn_rm, downsample_bn_rv;

            if (has_downsample) {
                downsample_conv_w = get_param(block_name + "_downsample_0_weight");
                downsample_bn_w = get_param(block_name + "_downsample_1_weight");
                downsample_bn_b = get_param(block_name + "_downsample_1_bias");
                downsample_bn_rm = get_param(block_name + "_downsample_1_running_mean");
                downsample_bn_rv = get_param(block_name + "_downsample_1_running_var");
            }

            x = basic_block_fn(
                x,
                get_param(block_name + "_conv1_weight"),
                get_param(block_name + "_bn1_weight"),
                get_param(block_name + "_bn1_bias"),
                get_param(block_name + "_bn1_running_mean"),
                get_param(block_name + "_bn1_running_var"),
                get_param(block_name + "_conv2_weight"),
                get_param(block_name + "_bn2_weight"),
                get_param(block_name + "_bn2_bias"),
                get_param(block_name + "_bn2_running_mean"),
                get_param(block_name + "_bn2_running_var"),
                downsample_conv_w,
                downsample_bn_w,
                downsample_bn_b,
                downsample_bn_rm,
                downsample_bn_rv,
                stride,
                is_training
            );
        }
    }

    x = torch::adaptive_avg_pool2d(x, {1, 1});
    x = x.view({x.size(0), -1});
    x = torch::linear(x, get_param("fc_weight"), get_param("fc_bias"));
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "ResNet18 forward function with aligned memory access (CPU)");
}
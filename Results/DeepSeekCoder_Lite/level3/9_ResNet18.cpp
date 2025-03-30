#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>

namespace py = pybind11;

#define BLOCK_SIZE 256

void fused_add_relu_aligned_kernel(
    const float* __restrict__ input,
    const float* __restrict__ identity,
    float* __restrict__ output,
    const int size
) {
    const int tid = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();
    const int stride = BLOCK_SIZE * num_threads;
    const int vec_size = size / 4;
    
    // Process 4 elements at a time using float4
    const float4* input4 = reinterpret_cast<const float4*>(input);
    const float4* identity4 = reinterpret_cast<const float4*>(identity);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    for (int i = tid; i < vec_size; i += num_threads) {
        // Use __builtin_assume_aligned for aligned memory access
        float4 in_val = *reinterpret_cast<const float4*>(&input4[i]);
        float4 id_val = *reinterpret_cast<const float4*>(&identity4[i]);
        
        float4 result;
        result.x = fmaxf(in_val.x + id_val.x, 0.0f);
        result.y = fmaxf(in_val.y + id_val.y, 0.0f);
        result.z = fmaxf(in_val.z + id_val.z, 0.0f);
        result.w = fmaxf(in_val.w + id_val.w, 0.0f);
        
        output4[i] = result;
    }
    
    // Handle remaining elements
    const int remaining_start = vec_size * 4;
    for (int i = remaining_start + tid; i < size; i += num_threads) {
        float in_val = input[i];
        float id_val = identity[i];
        output[i] = fmaxf(in_val + id_val, 0.0f);
    }
}

void launch_fused_kernel(torch::Tensor& x, const torch::Tensor& identity) {
    const int size = x.numel();
    const int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Ensure tensors are contiguous for aligned access
    auto x_contig = x.contiguous();
    auto identity_contig = identity.contiguous();
    
    fused_add_relu_aligned_kernel(
        x_contig.data_ptr<float>(),
        identity_contig.data_ptr<float>(),
        x_contig.data_ptr<float>(),
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

    // Ensure contiguous memory layout for convolutions
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
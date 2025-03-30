#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// C++ CPU implementation for ReLU activation
template <typename scalar_t>
void relu_cpu_kernel(const scalar_t* input, scalar_t* output, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        scalar_t x = input[i];
        output[i] = x > scalar_t(0) ? x : scalar_t(0);
    }
}

// Wrapper function for ReLU kernel
torch::Tensor relu_cpu(torch::Tensor input) {
    auto output = torch::empty_like(input);
    auto size = input.numel();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "relu_cpu", ([&] {
        relu_cpu_kernel<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

// Helper function to extract tensors from params
torch::Tensor get_tensor_from_params(py::object params, const std::string& key) {
    py::object tensor_obj = params.attr("__getitem__")(key);
    return tensor_obj.cast<torch::Tensor>().contiguous();
}

// Main forward function
torch::Tensor module_forward(
    torch::Tensor x,
    py::object params,
    bool is_training
) {
    // Ensure input is contiguous and on CPU
    x = x.contiguous();

    // Extract parameters
    torch::Tensor batchnorm_running_mean = get_tensor_from_params(params, "batchnorm_running_mean");
    torch::Tensor batchnorm_running_var = get_tensor_from_params(params, "batchnorm_running_var");
    torch::Tensor batchnorm_weight = get_tensor_from_params(params, "batchnorm_weight");
    torch::Tensor batchnorm_bias = get_tensor_from_params(params, "batchnorm_bias");
    torch::Tensor conv_weight = get_tensor_from_params(params, "conv_weight");

    // Batch Normalization
    x = at::batch_norm(
        x,
        batchnorm_weight,
        batchnorm_bias,
        batchnorm_running_mean,
        batchnorm_running_var,
        is_training,
        /*momentum=*/0.1,
        /*eps=*/1e-5,
        /*cudnn_enabled=*/false
    );

    // ReLU activation using custom CPU kernel
    x = relu_cpu(x);

    // Convolution
    x = at::conv2d(
        x,
        conv_weight,
        /*bias=*/c10::nullopt,
        /*stride=*/std::vector<int64_t>{1, 1},
        /*padding=*/std::vector<int64_t>{0, 0},
        /*dilation=*/std::vector<int64_t>{1, 1},
        /*groups=*/1
    );

    // Average Pooling
    x = at::avg_pool2d(
        x,
        /*kernel_size=*/std::vector<int64_t>{2, 2},
        /*stride=*/std::vector<int64_t>{2, 2},
        /*padding=*/std::vector<int64_t>{0, 0},
        /*ceil_mode=*/false,
        /*count_include_pad=*/false,
        /*divisor_override=*/c10::nullopt
    );

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_forward, "Module forward function");
}
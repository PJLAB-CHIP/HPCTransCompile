#include <torch/extension.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <c10/core/optional.h>
#include <omp.h>

namespace py = pybind11;

// Renamed the function to avoid conflict with PyTorch's built-in function
torch::Tensor custom_channel_shuffle(torch::Tensor x, int64_t groups) {
    int64_t batch_size = x.size(0);
    int64_t channels = x.size(1);
    int64_t height = x.size(2);
    int64_t width = x.size(3);

    int64_t channels_per_group = channels / groups;

    // Reshape
    x = x.view({batch_size, groups, channels_per_group, height, width});

    // Transpose
    x = x.permute({0, 2, 1, 3, 4}).contiguous();

    // Flatten
    x = x.view({batch_size, -1, height, width});

    return x;
}

torch::Tensor forward(torch::Tensor x, py::object params, bool is_training) {
    // First group conv + bn
    auto weight_conv1 = params.attr("__getitem__")("conv1_weight").cast<torch::Tensor>();
    auto groups_conv1 = params.attr("__getitem__")("groups").cast<int64_t>();

    c10::optional<torch::Tensor> bias_conv1 = c10::nullopt;

    std::vector<int64_t> stride1{1, 1};
    std::vector<int64_t> padding1{0, 0};
    std::vector<int64_t> dilation1{1, 1};

    auto out = torch::conv2d(
        x,
        weight_conv1,
        bias_conv1,
        stride1,
        padding1,
        dilation1,
        groups_conv1);

    // Batch Norm parameters
    auto weight_bn1 = params.attr("__getitem__")("bn1_weight").cast<torch::Tensor>();
    auto bias_bn1 = params.attr("__getitem__")("bn1_bias").cast<torch::Tensor>();
    auto running_mean_bn1 = params.attr("__getitem__")("bn1_running_mean").cast<torch::Tensor>();
    auto running_var_bn1 = params.attr("__getitem__")("bn1_running_var").cast<torch::Tensor>();

    out = torch::batch_norm(
        out,
        weight_bn1,
        bias_bn1,
        running_mean_bn1,
        running_var_bn1,
        is_training,
        /*momentum=*/0.1,
        /*eps=*/1e-5,
        /*cudnn_enabled=*/true);

    out = torch::relu(out);

    // Depthwise conv + bn
    auto weight_conv2 = params.attr("__getitem__")("conv2_weight").cast<torch::Tensor>();
    auto groups_conv2 = params.attr("__getitem__")("mid_channels").cast<int64_t>();

    c10::optional<torch::Tensor> bias_conv2 = c10::nullopt;

    std::vector<int64_t> stride2{1, 1};
    std::vector<int64_t> padding2{1, 1};
    std::vector<int64_t> dilation2{1, 1};

    out = torch::conv2d(
        out,
        weight_conv2,
        bias_conv2,
        stride2,
        padding2,
        dilation2,
        groups_conv2);

    // Batch Norm parameters
    auto weight_bn2 = params.attr("__getitem__")("bn2_weight").cast<torch::Tensor>();
    auto bias_bn2 = params.attr("__getitem__")("bn2_bias").cast<torch::Tensor>();
    auto running_mean_bn2 = params.attr("__getitem__")("bn2_running_mean").cast<torch::Tensor>();
    auto running_var_bn2 = params.attr("__getitem__")("bn2_running_var").cast<torch::Tensor>();

    out = torch::batch_norm(
        out,
        weight_bn2,
        bias_bn2,
        running_mean_bn2,
        running_var_bn2,
        is_training,
        /*momentum=*/0.1,
        /*eps=*/1e-5,
        /*cudnn_enabled=*/true);

    // Channel shuffle
    out = custom_channel_shuffle(out, groups_conv1);

    // Second group conv + bn
    auto weight_conv3 = params.attr("__getitem__")("conv3_weight").cast<torch::Tensor>();
    c10::optional<torch::Tensor> bias_conv3 = c10::nullopt;

    std::vector<int64_t> stride3{1, 1};
    std::vector<int64_t> padding3{0, 0};
    std::vector<int64_t> dilation3{1, 1};

    out = torch::conv2d(
        out,
        weight_conv3,
        bias_conv3,
        stride3,
        padding3,
        dilation3,
        groups_conv1);

    // Batch Norm parameters
    auto weight_bn3 = params.attr("__getitem__")("bn3_weight").cast<torch::Tensor>();
    auto bias_bn3 = params.attr("__getitem__")("bn3_bias").cast<torch::Tensor>();
    auto running_mean_bn3 = params.attr("__getitem__")("bn3_running_mean").cast<torch::Tensor>();
    auto running_var_bn3 = params.attr("__getitem__")("bn3_running_var").cast<torch::Tensor>();

    out = torch::batch_norm(
        out,
        weight_bn3,
        bias_bn3,
        running_mean_bn3,
        running_var_bn3,
        is_training,
        /*momentum=*/0.1,
        /*eps=*/1e-5,
        /*cudnn_enabled=*/true);

    out = torch::relu(out);

    // Shortcut
    torch::Tensor shortcut;
    if (py::hasattr(params, "__contains__") && params.attr("__contains__")("shortcut_conv_weight").cast<bool>()) {
        auto weight_shortcut_conv = params.attr("__getitem__")("shortcut_conv_weight").cast<torch::Tensor>();

        c10::optional<torch::Tensor> bias_shortcut_conv = c10::nullopt;

        std::vector<int64_t> stride_sc{1, 1};
        std::vector<int64_t> padding_sc{0, 0};
        std::vector<int64_t> dilation_sc{1, 1};

        shortcut = torch::conv2d(
            x,
            weight_shortcut_conv,
            bias_shortcut_conv,
            stride_sc,
            padding_sc,
            dilation_sc,
            /*groups=*/1);

        auto weight_bn_sc = params.attr("__getitem__")("shortcut_bn_weight").cast<torch::Tensor>();
        auto bias_bn_sc = params.attr("__getitem__")("shortcut_bn_bias").cast<torch::Tensor>();
        auto running_mean_bn_sc = params.attr("__getitem__")("shortcut_bn_running_mean").cast<torch::Tensor>();
        auto running_var_bn_sc = params.attr("__getitem__")("shortcut_bn_running_var").cast<torch::Tensor>();

        shortcut = torch::batch_norm(
            shortcut,
            weight_bn_sc,
            bias_bn_sc,
            running_mean_bn_sc,
            running_var_bn_sc,
            is_training,
            /*momentum=*/0.1,
            /*eps=*/1e-5,
            /*cudnn_enabled=*/true);
    } else {
        shortcut = x;
    }

    out += shortcut;

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ShuffleNet Unit forward");
}
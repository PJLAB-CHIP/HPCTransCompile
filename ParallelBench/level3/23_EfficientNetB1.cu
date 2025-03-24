#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This version expects a ParameterDict-like Python object for "params" rather than
// a standard Python dict. We retrieve each named parameter via __getitem__ so that
// any Python object implementing the mapping protocol (like ParameterDict) can be used.
//
// Usage from Python might look like:
//   cuda_fn = load(name="...", sources=["this_file.cu"], ... , with_cuda=True)
//   output = cuda_fn.forward(x, params, is_training)
//
// If x and params are on CUDA, the underlying ops will run on GPU.

namespace py = pybind11;

// -----------------------------------------------------------------------------
// MBConv block: expansion (1x1) -> depthwise (3x3) -> projection (1x1)
// -----------------------------------------------------------------------------
static torch::Tensor mbconv_block(
    torch::Tensor x,
    torch::Tensor conv1_w,
    torch::Tensor conv1_bn_w,
    torch::Tensor conv1_bn_b,
    torch::Tensor conv1_bn_rm,
    torch::Tensor conv1_bn_rv,
    torch::Tensor conv2_w,
    torch::Tensor conv2_bn_w,
    torch::Tensor conv2_bn_b,
    torch::Tensor conv2_bn_rm,
    torch::Tensor conv2_bn_rv,
    torch::Tensor conv3_w,
    torch::Tensor conv3_bn_w,
    torch::Tensor conv3_bn_b,
    torch::Tensor conv3_bn_rm,
    torch::Tensor conv3_bn_rv,
    int64_t stride,
    bool is_training
) {
    // 1) Expansion conv (1x1)
    x = at::conv2d(
        /*input=*/x,
        /*weight=*/conv1_w,
        /*bias=*/c10::nullopt,
        /*stride=*/torch::IntArrayRef({1, 1}),
        /*padding=*/torch::IntArrayRef({0, 0}),
        /*dilation=*/torch::IntArrayRef({1, 1}),
        /*groups=*/1
    );
    x = at::batch_norm(
        x,
        conv1_bn_w, conv1_bn_b,
        conv1_bn_rm, conv1_bn_rv,
        is_training,
        /*momentum=*/0.1,
        /*eps=*/1e-5,
        /*cudnn_enabled=*/true
    );
    x = x.clamp(0, 6);  // ReLU6

    // 2) Depthwise conv (3x3)
    x = at::conv2d(
        /*input=*/x,
        /*weight=*/conv2_w,
        /*bias=*/c10::nullopt,
        /*stride=*/torch::IntArrayRef({(int64_t)stride, (int64_t)stride}),
        /*padding=*/torch::IntArrayRef({1, 1}),
        /*dilation=*/torch::IntArrayRef({1, 1}),
        /*groups=*/conv2_w.size(0) // depthwise
    );
    x = at::batch_norm(
        x,
        conv2_bn_w, conv2_bn_b,
        conv2_bn_rm, conv2_bn_rv,
        is_training,
        0.1,
        1e-5,
        true
    );
    x = x.clamp(0, 6);  // ReLU6

    // 3) Projection conv (1x1)
    x = at::conv2d(
        /*input=*/x,
        /*weight=*/conv3_w,
        /*bias=*/c10::nullopt,
        /*stride=*/torch::IntArrayRef({1, 1}),
        /*padding=*/torch::IntArrayRef({0, 0}),
        /*dilation=*/torch::IntArrayRef({1, 1}),
        /*groups=*/1
    );
    x = at::batch_norm(
        x,
        conv3_bn_w, conv3_bn_b,
        conv3_bn_rm, conv3_bn_rv,
        is_training,
        0.1,
        1e-5,
        true
    );

    return x;
}

// -----------------------------------------------------------------------------
// Forward pass for EfficientNetB1 mirroring the reference PyTorch code:
//
//  1) Initial conv + BN + ReLU
//  2) 7 MBConv blocks
//  3) Final conv + BN + ReLU
//  4) AdaptiveAvgPool -> Flatten -> FC
//
// Here, "params" is a Python object that must support __getitem__ for each
// parameter name: e.g. params["conv1_w"], params["bn1_w"], etc.
// This accommodates PyTorch nn.ParameterDict objects.
// -----------------------------------------------------------------------------
torch::Tensor forward(
    torch::Tensor x,
    py::object params,  // Accept a generic Python object (e.g. ParameterDict)
    bool is_training
) {
    // 1) Initial conv, BN, ReLU
    auto conv1_w = params.attr("__getitem__")(py::str("conv1_w")).cast<torch::Tensor>();
    auto bn1_rm  = params.attr("__getitem__")(py::str("bn1_rm")).cast<torch::Tensor>();
    auto bn1_rv  = params.attr("__getitem__")(py::str("bn1_rv")).cast<torch::Tensor>();
    auto bn1_w   = params.attr("__getitem__")(py::str("bn1_w")).cast<torch::Tensor>();
    auto bn1_b   = params.attr("__getitem__")(py::str("bn1_b")).cast<torch::Tensor>();

    x = at::conv2d(
        /*input=*/x,
        /*weight=*/conv1_w,
        /*bias=*/c10::nullopt,
        /*stride=*/torch::IntArrayRef({2, 2}),
        /*padding=*/torch::IntArrayRef({1, 1}),
        /*dilation=*/torch::IntArrayRef({1, 1}),
        /*groups=*/1
    );
    x = at::batch_norm(
        x,
        bn1_w, bn1_b,
        bn1_rm, bn1_rv,
        is_training,
        0.1,
        1e-5,
        true
    );
    x = at::relu(x);

    // 2) MBConv blocks
    std::vector<int64_t> strides = {1, 2, 2, 2, 1, 2, 1};
    for (int i = 0; i < 7; ++i) {
        std::string prefix = "mbconv" + std::to_string(i + 1) + "_";

        auto conv1_w_ = params.attr("__getitem__")(py::str(prefix + "conv1_w")).cast<torch::Tensor>();
        auto conv1_bn_w_ = params.attr("__getitem__")(py::str(prefix + "conv1_bn_w")).cast<torch::Tensor>();
        auto conv1_bn_b_ = params.attr("__getitem__")(py::str(prefix + "conv1_bn_b")).cast<torch::Tensor>();
        auto conv1_bn_rm_ = params.attr("__getitem__")(py::str(prefix + "conv1_bn_rm")).cast<torch::Tensor>();
        auto conv1_bn_rv_ = params.attr("__getitem__")(py::str(prefix + "conv1_bn_rv")).cast<torch::Tensor>();

        auto conv2_w_ = params.attr("__getitem__")(py::str(prefix + "conv2_w")).cast<torch::Tensor>();
        auto conv2_bn_w_ = params.attr("__getitem__")(py::str(prefix + "conv2_bn_w")).cast<torch::Tensor>();
        auto conv2_bn_b_ = params.attr("__getitem__")(py::str(prefix + "conv2_bn_b")).cast<torch::Tensor>();
        auto conv2_bn_rm_ = params.attr("__getitem__")(py::str(prefix + "conv2_bn_rm")).cast<torch::Tensor>();
        auto conv2_bn_rv_ = params.attr("__getitem__")(py::str(prefix + "conv2_bn_rv")).cast<torch::Tensor>();

        auto conv3_w_ = params.attr("__getitem__")(py::str(prefix + "conv3_w")).cast<torch::Tensor>();
        auto conv3_bn_w_ = params.attr("__getitem__")(py::str(prefix + "conv3_bn_w")).cast<torch::Tensor>();
        auto conv3_bn_b_ = params.attr("__getitem__")(py::str(prefix + "conv3_bn_b")).cast<torch::Tensor>();
        auto conv3_bn_rm_ = params.attr("__getitem__")(py::str(prefix + "conv3_bn_rm")).cast<torch::Tensor>();
        auto conv3_bn_rv_ = params.attr("__getitem__")(py::str(prefix + "conv3_bn_rv")).cast<torch::Tensor>();

        x = mbconv_block(
            x,
            conv1_w_,
            conv1_bn_w_, conv1_bn_b_, conv1_bn_rm_, conv1_bn_rv_,
            conv2_w_,
            conv2_bn_w_, conv2_bn_b_, conv2_bn_rm_, conv2_bn_rv_,
            conv3_w_,
            conv3_bn_w_, conv3_bn_b_, conv3_bn_rm_, conv3_bn_rv_,
            strides[i],
            is_training
        );
    }

    // 3) Final conv + BN + ReLU
    auto conv2_w = params.attr("__getitem__")(py::str("conv2_w")).cast<torch::Tensor>();
    auto bn2_rm  = params.attr("__getitem__")(py::str("bn2_rm")).cast<torch::Tensor>();
    auto bn2_rv  = params.attr("__getitem__")(py::str("bn2_rv")).cast<torch::Tensor>();
    auto bn2_w   = params.attr("__getitem__")(py::str("bn2_w")).cast<torch::Tensor>();
    auto bn2_b   = params.attr("__getitem__")(py::str("bn2_b")).cast<torch::Tensor>();

    x = at::conv2d(
        /*input=*/x,
        /*weight=*/conv2_w,
        /*bias=*/c10::nullopt,
        /*stride=*/torch::IntArrayRef({1, 1}),
        /*padding=*/torch::IntArrayRef({0, 0}),
        /*dilation=*/torch::IntArrayRef({1, 1}),
        /*groups=*/1
    );
    x = at::batch_norm(
        x,
        bn2_w, bn2_b,
        bn2_rm, bn2_rv,
        is_training,
        0.1,
        1e-5,
        true
    );
    x = at::relu(x);

    // 4) Adaptive average pool -> Flatten -> FC
    x = at::adaptive_avg_pool2d(x, torch::IntArrayRef({1, 1}));
    x = x.view({x.size(0), -1});

    auto fc_w = params.attr("__getitem__")(py::str("fc_w")).cast<torch::Tensor>();
    auto fc_b = params.attr("__getitem__")(py::str("fc_b")).cast<torch::Tensor>();
    x = at::matmul(x, fc_w.t()) + fc_b;

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward,
        "EfficientNetB1 forward pass (CUDA/C++) using a ParameterDict-like object."
    );
}
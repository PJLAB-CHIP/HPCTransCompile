#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/*
  A helper function that replicates the "double_conv_fn" from Python:
    1) Conv2D -> BatchNorm -> Softmax  (first pair)
    2) Conv2D -> BatchNorm -> Softmax  (second pair)
*/
at::Tensor double_conv_fn(
    const at::Tensor& x_in,
    const at::Tensor& conv1_w,
    const at::Tensor& conv1_b,
    const at::Tensor& bn1_mean,
    const at::Tensor& bn1_var,
    const at::Tensor& bn1_w,
    const at::Tensor& bn1_b,
    const at::Tensor& conv2_w,
    const at::Tensor& conv2_b,
    const at::Tensor& bn2_mean,
    const at::Tensor& bn2_var,
    const at::Tensor& bn2_w,
    const at::Tensor& bn2_b,
    bool is_training
) {
  // First conv + BN + softmax
  auto x = at::conv2d(
      x_in, conv1_w, conv1_b, /*stride=*/{1,1}, /*padding=*/{1,1});
  x = at::batch_norm(
      x,
      bn1_w,
      bn1_b,
      bn1_mean,
      bn1_var,
      is_training,
      /*momentum=*/0.1,
      /*eps=*/1e-5,
      /*cudnn_enabled=*/true
  );
  // softmax along last dimension (dim=3 for NCHW)
  x = at::softmax(x, /*dim=*/3);

  // Second conv + BN + softmax
  x = at::conv2d(
      x, conv2_w, conv2_b, /*stride=*/{1,1}, /*padding=*/{1,1});
  x = at::batch_norm(
      x,
      bn2_w,
      bn2_b,
      bn2_mean,
      bn2_var,
      is_training,
      /*momentum=*/0.1,
      /*eps=*/1e-5,
      /*cudnn_enabled=*/true
  );
  x = at::softmax(x, /*dim=*/3);

  return x;
}

/*
  A function that replicates the entire UNet forward pass described in the
  given PyTorch code. We accept:
    - x            : input tensor
    - param_dict   : a Python object that can be a ParameterDict or dict
    - is_training  : bool
*/
at::Tensor forward_unet(
    const at::Tensor& x,
    py::object param_dict,
    bool is_training
) {
  // Helper lambda to fetch a tensor from param_dict (ParameterDict or dict).
  auto get_param = [&](const std::string& key) {
    return param_dict.attr("__getitem__")(py::str(key)).cast<at::Tensor>();
  };

  // ----------------------------------------------------------------------------
  // Encoder path
  // ----------------------------------------------------------------------------
  auto enc1 = double_conv_fn(
      x,
      get_param("enc1_conv1_w"),
      get_param("enc1_conv1_b"),
      get_param("enc1_bn1_mean"),
      get_param("enc1_bn1_var"),
      get_param("enc1_bn1_w"),
      get_param("enc1_bn1_b"),
      get_param("enc1_conv2_w"),
      get_param("enc1_conv2_b"),
      get_param("enc1_bn2_mean"),
      get_param("enc1_bn2_var"),
      get_param("enc1_bn2_w"),
      get_param("enc1_bn2_b"),
      is_training
  );
  auto p1 = at::max_pool2d(enc1, {2, 2}, {2, 2});

  auto enc2 = double_conv_fn(
      p1,
      get_param("enc2_conv1_w"),
      get_param("enc2_conv1_b"),
      get_param("enc2_bn1_mean"),
      get_param("enc2_bn1_var"),
      get_param("enc2_bn1_w"),
      get_param("enc2_bn1_b"),
      get_param("enc2_conv2_w"),
      get_param("enc2_conv2_b"),
      get_param("enc2_bn2_mean"),
      get_param("enc2_bn2_var"),
      get_param("enc2_bn2_w"),
      get_param("enc2_bn2_b"),
      is_training
  );
  auto p2 = at::max_pool2d(enc2, {2, 2}, {2, 2});

  auto enc3 = double_conv_fn(
      p2,
      get_param("enc3_conv1_w"),
      get_param("enc3_conv1_b"),
      get_param("enc3_bn1_mean"),
      get_param("enc3_bn1_var"),
      get_param("enc3_bn1_w"),
      get_param("enc3_bn1_b"),
      get_param("enc3_conv2_w"),
      get_param("enc3_conv2_b"),
      get_param("enc3_bn2_mean"),
      get_param("enc3_bn2_var"),
      get_param("enc3_bn2_w"),
      get_param("enc3_bn2_b"),
      is_training
  );
  auto p3 = at::max_pool2d(enc3, {2, 2}, {2, 2});

  auto enc4 = double_conv_fn(
      p3,
      get_param("enc4_conv1_w"),
      get_param("enc4_conv1_b"),
      get_param("enc4_bn1_mean"),
      get_param("enc4_bn1_var"),
      get_param("enc4_bn1_w"),
      get_param("enc4_bn1_b"),
      get_param("enc4_conv2_w"),
      get_param("enc4_conv2_b"),
      get_param("enc4_bn2_mean"),
      get_param("enc4_bn2_var"),
      get_param("enc4_bn2_w"),
      get_param("enc4_bn2_b"),
      is_training
  );
  auto p4 = at::max_pool2d(enc4, {2, 2}, {2, 2});

  // ----------------------------------------------------------------------------
  // Bottleneck
  // ----------------------------------------------------------------------------
  auto bottleneck = double_conv_fn(
      p4,
      get_param("bottleneck_conv1_w"),
      get_param("bottleneck_conv1_b"),
      get_param("bottleneck_bn1_mean"),
      get_param("bottleneck_bn1_var"),
      get_param("bottleneck_bn1_w"),
      get_param("bottleneck_bn1_b"),
      get_param("bottleneck_conv2_w"),
      get_param("bottleneck_conv2_b"),
      get_param("bottleneck_bn2_mean"),
      get_param("bottleneck_bn2_var"),
      get_param("bottleneck_bn2_w"),
      get_param("bottleneck_bn2_b"),
      is_training
  );

  // ----------------------------------------------------------------------------
  // Decoder path
  // ----------------------------------------------------------------------------
  auto d4 = at::conv_transpose2d(
      bottleneck,
      get_param("upconv4_w"),
      get_param("upconv4_b"),
      /*stride=*/{2, 2}
  );
  d4 = at::cat({d4, enc4}, /*dim=*/1);
  d4 = double_conv_fn(
      d4,
      get_param("dec4_conv1_w"),
      get_param("dec4_conv1_b"),
      get_param("dec4_bn1_mean"),
      get_param("dec4_bn1_var"),
      get_param("dec4_bn1_w"),
      get_param("dec4_bn1_b"),
      get_param("dec4_conv2_w"),
      get_param("dec4_conv2_b"),
      get_param("dec4_bn2_mean"),
      get_param("dec4_bn2_var"),
      get_param("dec4_bn2_w"),
      get_param("dec4_bn2_b"),
      is_training
  );

  auto d3 = at::conv_transpose2d(
      d4,
      get_param("upconv3_w"),
      get_param("upconv3_b"),
      /*stride=*/{2, 2}
  );
  d3 = at::cat({d3, enc3}, /*dim=*/1);
  d3 = double_conv_fn(
      d3,
      get_param("dec3_conv1_w"),
      get_param("dec3_conv1_b"),
      get_param("dec3_bn1_mean"),
      get_param("dec3_bn1_var"),
      get_param("dec3_bn1_w"),
      get_param("dec3_bn1_b"),
      get_param("dec3_conv2_w"),
      get_param("dec3_conv2_b"),
      get_param("dec3_bn2_mean"),
      get_param("dec3_bn2_var"),
      get_param("dec3_bn2_w"),
      get_param("dec3_bn2_b"),
      is_training
  );

  auto d2 = at::conv_transpose2d(
      d3,
      get_param("upconv2_w"),
      get_param("upconv2_b"),
      /*stride=*/{2, 2}
  );
  d2 = at::cat({d2, enc2}, /*dim=*/1);
  d2 = double_conv_fn(
      d2,
      get_param("dec2_conv1_w"),
      get_param("dec2_conv1_b"),
      get_param("dec2_bn1_mean"),
      get_param("dec2_bn1_var"),
      get_param("dec2_bn1_w"),
      get_param("dec2_bn1_b"),
      get_param("dec2_conv2_w"),
      get_param("dec2_conv2_b"),
      get_param("dec2_bn2_mean"),
      get_param("dec2_bn2_var"),
      get_param("dec2_bn2_w"),
      get_param("dec2_bn2_b"),
      is_training
  );

  auto d1 = at::conv_transpose2d(
      d2,
      get_param("upconv1_w"),
      get_param("upconv1_b"),
      /*stride=*/{2, 2}
  );
  d1 = at::cat({d1, enc1}, /*dim=*/1);
  d1 = double_conv_fn(
      d1,
      get_param("dec1_conv1_w"),
      get_param("dec1_conv1_b"),
      get_param("dec1_bn1_mean"),
      get_param("dec1_bn1_var"),
      get_param("dec1_bn1_w"),
      get_param("dec1_bn1_b"),
      get_param("dec1_conv2_w"),
      get_param("dec1_conv2_b"),
      get_param("dec1_bn2_mean"),
      get_param("dec1_bn2_var"),
      get_param("dec1_bn2_w"),
      get_param("dec1_bn2_b"),
      is_training
  );

  // final conv
  auto output = at::conv2d(
      d1,
      get_param("final_conv_w"),
      get_param("final_conv_b")
  );

  return output;
}

// Create the PyBind11 module. The name "TORCH_EXTENSION_NAME" is a
// special macro used by PyTorch for C++ extensions.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "forward",
    &forward_unet,
    "UNet forward pass (CUDA) that accepts (Tensor, ParameterDict/dict, bool)."
  );
}
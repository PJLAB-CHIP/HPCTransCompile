import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, params: dict, is_training: bool) -> torch.Tensor:
    """
    Functional version of the UNet with softmax activation

    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        params: Dictionary of parameters
        is_training: Boolean indicating if the model is in training mode

    Returns:
        Output tensor of shape (batch_size, out_channels, height, width)
    """

    def double_conv_fn(
        x,
        conv1_weight,
        conv1_bias,
        bn1_weight,
        bn1_bias,
        bn1_mean,
        bn1_var,
        conv2_weight,
        conv2_bias,
        bn2_weight,
        bn2_bias,
        bn2_mean,
        bn2_var,
        is_training,
    ):
        x = F.conv2d(x, conv1_weight, conv1_bias, padding=1)
        x = F.batch_norm(x, bn1_mean, bn1_var, bn1_weight, bn1_bias, is_training)
        x = F.softmax(x, dim=-1)
        x = F.conv2d(x, conv2_weight, conv2_bias, padding=1)
        x = F.batch_norm(x, bn2_mean, bn2_var, bn2_weight, bn2_bias, is_training)
        x = F.softmax(x, dim=-1)
        return x

    # Encoder path
    enc1 = double_conv_fn(
        x,
        params["enc1_conv1_w"],
        params["enc1_conv1_b"],
        params["enc1_bn1_w"],
        params["enc1_bn1_b"],
        params["enc1_bn1_mean"],
        params["enc1_bn1_var"],
        params["enc1_conv2_w"],
        params["enc1_conv2_b"],
        params["enc1_bn2_w"],
        params["enc1_bn2_b"],
        params["enc1_bn2_mean"],
        params["enc1_bn2_var"],
        is_training,
    )

    p1 = F.max_pool2d(enc1, kernel_size=2, stride=2)
    enc2 = double_conv_fn(
        p1,
        params["enc2_conv1_w"],
        params["enc2_conv1_b"],
        params["enc2_bn1_w"],
        params["enc2_bn1_b"],
        params["enc2_bn1_mean"],
        params["enc2_bn1_var"],
        params["enc2_conv2_w"],
        params["enc2_conv2_b"],
        params["enc2_bn2_w"],
        params["enc2_bn2_b"],
        params["enc2_bn2_mean"],
        params["enc2_bn2_var"],
        is_training,
    )

    p2 = F.max_pool2d(enc2, kernel_size=2, stride=2)
    enc3 = double_conv_fn(
        p2,
        params["enc3_conv1_w"],
        params["enc3_conv1_b"],
        params["enc3_bn1_w"],
        params["enc3_bn1_b"],
        params["enc3_bn1_mean"],
        params["enc3_bn1_var"],
        params["enc3_conv2_w"],
        params["enc3_conv2_b"],
        params["enc3_bn2_w"],
        params["enc3_bn2_b"],
        params["enc3_bn2_mean"],
        params["enc3_bn2_var"],
        is_training,
    )

    p3 = F.max_pool2d(enc3, kernel_size=2, stride=2)
    enc4 = double_conv_fn(
        p3,
        params["enc4_conv1_w"],
        params["enc4_conv1_b"],
        params["enc4_bn1_w"],
        params["enc4_bn1_b"],
        params["enc4_bn1_mean"],
        params["enc4_bn1_var"],
        params["enc4_conv2_w"],
        params["enc4_conv2_b"],
        params["enc4_bn2_w"],
        params["enc4_bn2_b"],
        params["enc4_bn2_mean"],
        params["enc4_bn2_var"],
        is_training,
    )

    p4 = F.max_pool2d(enc4, kernel_size=2, stride=2)
    bottleneck = double_conv_fn(
        p4,
        params["bottleneck_conv1_w"],
        params["bottleneck_conv1_b"],
        params["bottleneck_bn1_w"],
        params["bottleneck_bn1_b"],
        params["bottleneck_bn1_mean"],
        params["bottleneck_bn1_var"],
        params["bottleneck_conv2_w"],
        params["bottleneck_conv2_b"],
        params["bottleneck_bn2_w"],
        params["bottleneck_bn2_b"],
        params["bottleneck_bn2_mean"],
        params["bottleneck_bn2_var"],
        is_training,
    )

    # Decoder path
    d4 = F.conv_transpose2d(
        bottleneck, params["upconv4_w"], params["upconv4_b"], stride=2
    )
    d4 = torch.cat([d4, enc4], dim=1)
    d4 = double_conv_fn(
        d4,
        params["dec4_conv1_w"],
        params["dec4_conv1_b"],
        params["dec4_bn1_w"],
        params["dec4_bn1_b"],
        params["dec4_bn1_mean"],
        params["dec4_bn1_var"],
        params["dec4_conv2_w"],
        params["dec4_conv2_b"],
        params["dec4_bn2_w"],
        params["dec4_bn2_b"],
        params["dec4_bn2_mean"],
        params["dec4_bn2_var"],
        is_training,
    )

    d3 = F.conv_transpose2d(d4, params["upconv3_w"], params["upconv3_b"], stride=2)
    d3 = torch.cat([d3, enc3], dim=1)
    d3 = double_conv_fn(
        d3,
        params["dec3_conv1_w"],
        params["dec3_conv1_b"],
        params["dec3_bn1_w"],
        params["dec3_bn1_b"],
        params["dec3_bn1_mean"],
        params["dec3_bn1_var"],
        params["dec3_conv2_w"],
        params["dec3_conv2_b"],
        params["dec3_bn2_w"],
        params["dec3_bn2_b"],
        params["dec3_bn2_mean"],
        params["dec3_bn2_var"],
        is_training,
    )

    d2 = F.conv_transpose2d(d3, params["upconv2_w"], params["upconv2_b"], stride=2)
    d2 = torch.cat([d2, enc2], dim=1)
    d2 = double_conv_fn(
        d2,
        params["dec2_conv1_w"],
        params["dec2_conv1_b"],
        params["dec2_bn1_w"],
        params["dec2_bn1_b"],
        params["dec2_bn1_mean"],
        params["dec2_bn1_var"],
        params["dec2_conv2_w"],
        params["dec2_conv2_b"],
        params["dec2_bn2_w"],
        params["dec2_bn2_b"],
        params["dec2_bn2_mean"],
        params["dec2_bn2_var"],
        is_training,
    )

    d1 = F.conv_transpose2d(d2, params["upconv1_w"], params["upconv1_b"], stride=2)
    d1 = torch.cat([d1, enc1], dim=1)
    d1 = double_conv_fn(
        d1,
        params["dec1_conv1_w"],
        params["dec1_conv1_b"],
        params["dec1_bn1_w"],
        params["dec1_bn1_b"],
        params["dec1_bn1_mean"],
        params["dec1_bn1_var"],
        params["dec1_conv2_w"],
        params["dec1_conv2_b"],
        params["dec1_bn2_w"],
        params["dec1_bn2_b"],
        params["dec1_bn2_mean"],
        params["dec1_bn2_var"],
        is_training,
    )

    return F.conv2d(d1, params["final_conv_w"], params["final_conv_b"])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(Model, self).__init__()

        self.params = nn.ParameterDict()

        # Initialize encoder parameters
        def init_double_conv(prefix, in_ch, out_ch):
            conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            bn1 = nn.BatchNorm2d(out_ch)
            conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
            bn2 = nn.BatchNorm2d(out_ch)

            self.params[f"{prefix}_conv1_w"] = nn.Parameter(conv1.weight.data.clone())
            self.params[f"{prefix}_conv1_b"] = nn.Parameter(conv1.bias.data.clone())
            self.params[f"{prefix}_bn1_w"] = nn.Parameter(bn1.weight.data.clone())
            self.params[f"{prefix}_bn1_b"] = nn.Parameter(bn1.bias.data.clone())
            self.params[f"{prefix}_bn1_mean"] = nn.Parameter(
                bn1.running_mean.data.clone()
            )
            self.params[f"{prefix}_bn1_var"] = nn.Parameter(
                bn1.running_var.data.clone()
            )

            self.params[f"{prefix}_conv2_w"] = nn.Parameter(conv2.weight.data.clone())
            self.params[f"{prefix}_conv2_b"] = nn.Parameter(conv2.bias.data.clone())
            self.params[f"{prefix}_bn2_w"] = nn.Parameter(bn2.weight.data.clone())
            self.params[f"{prefix}_bn2_b"] = nn.Parameter(bn2.bias.data.clone())
            self.params[f"{prefix}_bn2_mean"] = nn.Parameter(
                bn2.running_mean.data.clone()
            )
            self.params[f"{prefix}_bn2_var"] = nn.Parameter(
                bn2.running_var.data.clone()
            )

        init_double_conv("enc1", in_channels, features)
        init_double_conv("enc2", features, features * 2)
        init_double_conv("enc3", features * 2, features * 4)
        init_double_conv("enc4", features * 4, features * 8)
        init_double_conv("bottleneck", features * 8, features * 16)

        # Initialize decoder parameters
        def init_upconv(prefix, in_ch, out_ch):
            upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.params[f"{prefix}_w"] = nn.Parameter(upconv.weight.data.clone())
            self.params[f"{prefix}_b"] = nn.Parameter(upconv.bias.data.clone())

        init_upconv("upconv4", features * 16, features * 8)
        init_double_conv("dec4", features * 16, features * 8)
        init_upconv("upconv3", features * 8, features * 4)
        init_double_conv("dec3", features * 8, features * 4)
        init_upconv("upconv2", features * 4, features * 2)
        init_double_conv("dec2", features * 4, features * 2)
        init_upconv("upconv1", features * 2, features)
        init_double_conv("dec1", features * 2, features)

        final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.params["final_conv_w"] = nn.Parameter(final_conv.weight.data.clone())
        self.params["final_conv_b"] = nn.Parameter(final_conv.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


# First define the batch size and other configurations at module level
batch_size = 8
in_channels = 8
out_channels = 4
height = 64
width = 512
features = 64


# Define the test functions at module level
def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, features]

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace py = pybind11;

// Helper function to construct dictionary keys
std::string make_key(const std::string& base, const std::string& prefix) {
    return base + "_" + prefix;
}

// mlp_forward: Multi-layer perceptron forward pass
torch::Tensor mlp_forward(
    torch::Tensor x,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias,
    double drop = 0.0
) {
    x = torch::nn::functional::linear(x, fc1_weight, fc1_bias);
    x = torch::gelu(x);
    x = torch::nn::functional::dropout(
        x, torch::nn::functional::DropoutFuncOptions().p(drop).training(true)
    );
    x = torch::nn::functional::linear(x, fc2_weight, fc2_bias);
    x = torch::nn::functional::dropout(
        x, torch::nn::functional::DropoutFuncOptions().p(drop).training(true)
    );
    return x;
}

// window_partition: Partition tensor into windows
torch::Tensor window_partition(torch::Tensor x, int window_size) {
    auto sizes = x.sizes();
    int64_t B = sizes[0];
    int64_t H = sizes[1];
    int64_t W = sizes[2];
    int64_t C = sizes[3];
    x = x.view({B, H / window_size, window_size, W / window_size, window_size, C});
    x = x.permute({0, 1, 3, 2, 4, 5}).contiguous();
    x = x.view({-1, window_size, window_size, C});
    return x;
}

// window_reverse: Reconstruct tensor from windows
torch::Tensor window_reverse(
    torch::Tensor windows,
    int window_size,
    int H,
    int W
) {
    int64_t B = static_cast<int64_t>(windows.size(0) / (H * W / window_size / window_size));
    torch::Tensor x = windows.view({B, H / window_size, W / window_size, window_size, window_size, -1});
    x = x.permute({0, 1, 3, 2, 4, 5}).contiguous();
    x = x.view({B, H, W, -1});
    return x;
}

// window_attention_forward: Window-based attention forward pass
torch::Tensor window_attention_forward(
    torch::Tensor x,
    torch::Tensor qkv_weight,
    const c10::optional<torch::Tensor>& q_bias,
    const c10::optional<torch::Tensor>& v_bias,
    torch::Tensor proj_weight,
    torch::Tensor proj_bias,
    torch::Tensor logit_scale,
    const std::vector<torch::Tensor>& cpb_mlp_weights,
    const std::vector<torch::Tensor>& cpb_mlp_biases,
    torch::Tensor relative_coords_table,
    torch::Tensor relative_position_index,
    std::pair<int, int> window_size,
    int num_heads,
    const c10::optional<torch::Tensor>& mask = c10::nullopt,
    double attn_drop = 0.0,
    double proj_drop = 0.0
) {
    auto sizes = x.sizes();
    int64_t B_ = sizes[0];
    int64_t N = sizes[1];
    int64_t C = sizes[2];

    // Compute qkv bias if present
    c10::optional<torch::Tensor> qkv_bias = c10::nullopt;
    if (q_bias.has_value() && v_bias.has_value()) {
        torch::Tensor zeros = torch::zeros_like(*v_bias);
        qkv_bias = torch::cat({*q_bias, zeros, *v_bias});
    }

    // Linear transformation for Q, K, V
    torch::Tensor qkv = torch::nn::functional::linear(x, qkv_weight, qkv_bias.value());
    qkv = qkv.reshape({B_, N, 3, num_heads, -1}).permute({2, 0, 3, 1, 4});
    torch::Tensor q = qkv.select(0, 0);
    torch::Tensor k = qkv.select(0, 1);
    torch::Tensor v = qkv.select(0, 2);

    // Cosine attention
    torch::Tensor attn = torch::matmul(
        torch::nn::functional::normalize(q, torch::nn::functional::NormalizeFuncOptions().dim(-1)),
        torch::nn::functional::normalize(k, torch::nn::functional::NormalizeFuncOptions().dim(-1)).transpose(-2, -1));
    torch::Tensor logit_scale_clamped = torch::clamp(
        logit_scale,
        c10::nullopt,
        torch::log(torch::tensor(1.0 / 0.01, x.options()))
    ).exp();
    attn = attn * logit_scale_clamped;

    // Apply MLP to relative coordinates
    torch::Tensor x_pos = relative_coords_table;
    for (size_t i = 0; i < cpb_mlp_weights.size(); ++i) {
        if (i < cpb_mlp_weights.size() - 1) {
            x_pos = torch::relu(torch::nn::functional::linear(x_pos, cpb_mlp_weights[i], cpb_mlp_biases[i]));
        } else {
            x_pos = torch::nn::functional::linear(x_pos, cpb_mlp_weights[i], cpb_mlp_biases[i]);
        }
    }

    // Compute relative position bias
    torch::Tensor relative_position_bias_table = x_pos.view({-1, num_heads});
    // torch::tensor relative_position_bias = relative_position_bias_table.index(
    //     relative_position_index.to(torch::kint).view(-1)
    // ).view({window_size.first * window_size.second, window_size.first * window_size.second, -1});
    torch::Tensor index_tensor = relative_position_index.to(torch::kInt).view(-1);
    std::vector<at::indexing::TensorIndex> indices = {at::indexing::TensorIndex(index_tensor)};
    torch::Tensor relative_position_bias = relative_position_bias_table.index(indices)
        .view({window_size.first * window_size.second, window_size.first * window_size.second, -1});

    relative_position_bias = relative_position_bias.permute({2, 0, 1}).contiguous();
    relative_position_bias = 16 * torch::sigmoid(relative_position_bias);
    attn = attn + relative_position_bias.unsqueeze(0);

    // Apply mask if present
    if (mask.has_value()) {
        int64_t nW = mask->size(0);
        attn = attn.view({B_ / nW, nW, num_heads, N, N}) + mask->unsqueeze(1).unsqueeze(0);
        attn = attn.view({-1, num_heads, N, N});
        attn = torch::nn::functional::softmax(attn, torch::nn::functional::SoftmaxFuncOptions(-1));
    } else {
        attn = torch::nn::functional::softmax(attn, torch::nn::functional::SoftmaxFuncOptions(-1));
    }

    attn = torch::nn::functional::dropout(
        attn, torch::nn::functional::DropoutFuncOptions().p(attn_drop).training(true)
    );

    x = torch::matmul(attn, v).transpose(1, 2).reshape({B_, N, C});
    x = torch::nn::functional::linear(x, proj_weight, proj_bias);
    x = torch::nn::functional::dropout(
        x, torch::nn::functional::DropoutFuncOptions().p(proj_drop).training(true)
    );

    return x;
}

// swin_transformer_block_forward: Swin Transformer block forward pass
torch::Tensor swin_transformer_block_forward(
    torch::Tensor x,
    py::dict param_dict,
    py::dict buffer_dict,
    int i_layer,
    int i_block
) {
    std::string prefix = "layer" + std::to_string(i_layer) + "_block" + std::to_string(i_block);

    // Retrieve normalization parameters
    torch::Tensor norm1_weight = param_dict[py::str(make_key("norm1_weight", prefix))].cast<torch::Tensor>();
    torch::Tensor norm1_bias = param_dict[py::str(make_key("norm1_bias", prefix))].cast<torch::Tensor>();
    torch::Tensor norm2_weight = param_dict[py::str(make_key("norm2_weight", prefix))].cast<torch::Tensor>();
    torch::Tensor norm2_bias = param_dict[py::str(make_key("norm2_bias", prefix))].cast<torch::Tensor>();

    // Retrieve buffers
    torch::Tensor input_resolution_tensor = buffer_dict[py::str("input_resolution_layer" + std::to_string(i_layer))].cast<torch::Tensor>();
    // auto input_resolution = input_resolution_tensor.accessor<long, 1>();
    // int H = input_resolution[0];
    // int W = input_resolution[1];
    auto H = input_resolution_tensor[0].item<int>();
    auto W = input_resolution_tensor[1].item<int>();
    int window_size = buffer_dict[py::str("window_size")].cast<torch::Tensor>().item<int>();
    int shift_size = buffer_dict[py::str(make_key("shift_size", prefix))].cast<torch::Tensor>().item<int>();
    c10::optional<torch::Tensor> attn_mask = c10::nullopt;
    if (buffer_dict.contains(py::str(make_key("attn_mask", prefix)))) {
        attn_mask = buffer_dict[py::str(make_key("attn_mask", prefix))].cast<torch::Tensor>();
    }
    double drop_path_rate = buffer_dict[py::str(make_key("drop_path_rate", prefix))].cast<torch::Tensor>().item<double>();

    // Retrieve attention parameters
    torch::Tensor qkv_weight = param_dict[py::str(make_key("qkv_weight", prefix))].cast<torch::Tensor>();
    c10::optional<torch::Tensor> q_bias = c10::nullopt;
    c10::optional<torch::Tensor> v_bias = c10::nullopt;
    if (param_dict.contains(py::str(make_key("q_bias", prefix)))) {
        q_bias = param_dict[py::str(make_key("q_bias", prefix))].cast<torch::Tensor>();
    }
    if (param_dict.contains(py::str(make_key("v_bias", prefix)))) {
        v_bias = param_dict[py::str(make_key("v_bias", prefix))].cast<torch::Tensor>();
    }
    torch::Tensor proj_weight = param_dict[py::str(make_key("proj_weight", prefix))].cast<torch::Tensor>();
    torch::Tensor proj_bias = param_dict[py::str(make_key("proj_bias", prefix))].cast<torch::Tensor>();
    torch::Tensor logit_scale = param_dict[py::str(make_key("logit_scale", prefix))].cast<torch::Tensor>();
    std::vector<torch::Tensor> cpb_mlp_weights = {
        param_dict[py::str(make_key("cpb_mlp_weight0", prefix))].cast<torch::Tensor>(),
        param_dict[py::str(make_key("cpb_mlp_weight1", prefix))].cast<torch::Tensor>()
    };
    std::vector<torch::Tensor> cpb_mlp_biases = {
        param_dict[py::str(make_key("cpb_mlp_bias0", prefix))].cast<torch::Tensor>(),
        torch::Tensor()  // Second bias is None
    };
    torch::Tensor relative_coords_table = buffer_dict[py::str(make_key("relative_coords_table", prefix))].cast<torch::Tensor>();
    torch::Tensor relative_position_index = buffer_dict[py::str(make_key("relative_position_index", prefix))].cast<torch::Tensor>();
    torch::Tensor num_heads_tensor = buffer_dict[py::str("num_heads")].cast<torch::Tensor>();
    int num_heads = num_heads_tensor[i_layer].item<int>();
    double attn_drop = buffer_dict[py::str("attn_drop_rate")].cast<torch::Tensor>().item<double>();
    double proj_drop = buffer_dict[py::str("drop_rate")].cast<torch::Tensor>().item<double>();

    // Retrieve MLP parameters
    torch::Tensor fc1_weight = param_dict[py::str(make_key("fc1_weight", prefix))].cast<torch::Tensor>();
    torch::Tensor fc1_bias = param_dict[py::str(make_key("fc1_bias", prefix))].cast<torch::Tensor>();
    torch::Tensor fc2_weight = param_dict[py::str(make_key("fc2_weight", prefix))].cast<torch::Tensor>();
    torch::Tensor fc2_bias = param_dict[py::str(make_key("fc2_bias", prefix))].cast<torch::Tensor>();
    double drop = buffer_dict[py::str("drop_rate")].cast<torch::Tensor>().item<double>();

    // Implementation
    auto sizes = x.sizes();
    int64_t B = sizes[0];
    int64_t L = sizes[1];
    int64_t C = sizes[2];
    if (L != H * W) {
        throw std::runtime_error("input feature has wrong size");
    }

    torch::Tensor shortcut = x;
    x = torch::layer_norm(x, {C}, norm1_weight, norm1_bias);
    x = x.view({B, H, W, C});

    torch::Tensor shifted_x = x;
    if (shift_size > 0) {
        shifted_x = torch::roll(x, {-shift_size, -shift_size}, {1, 2});
    }

    torch::Tensor x_windows = window_partition(shifted_x, window_size);
    x_windows = x_windows.view({-1, window_size * window_size, C});

    torch::Tensor attn_windows = window_attention_forward(
        x_windows,
        qkv_weight,
        q_bias,
        v_bias,
        proj_weight,
        proj_bias,
        logit_scale,
        cpb_mlp_weights,
        cpb_mlp_biases,
        relative_coords_table,
        relative_position_index,
        {window_size, window_size},
        num_heads,
        attn_mask,
        attn_drop,
        proj_drop
    );

    attn_windows = attn_windows.view({-1, window_size, window_size, C});
    shifted_x = window_reverse(attn_windows, window_size, H, W);

    if (shift_size > 0) {
        x = torch::roll(shifted_x, {shift_size, shift_size}, {1, 2});
    } else {
        x = shifted_x;
    }
    x = x.view({B, H * W, C});

    if (drop_path_rate > 0.0) {
        double keep_prob = 1.0 - drop_path_rate;
        torch::Tensor mask = torch::rand({B, 1, 1}, x.options()) >= drop_path_rate;
        x = x / keep_prob * mask;
    }

    x = shortcut + x;

    shortcut = x;
    x = torch::layer_norm(x, {C}, norm2_weight, norm2_bias);
    x = mlp_forward(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, drop);

    if (drop_path_rate > 0.0) {
        double keep_prob = 1.0 - drop_path_rate;
        torch::Tensor mask = torch::rand({B, 1, 1}, x.options()) >= drop_path_rate;
        x = x / keep_prob * mask;
    }

    x = shortcut + x;

    return x;
}

// patch_merging_forward: Patch merging for downsampling
torch::Tensor patch_merging_forward(
    torch::Tensor x,
    py::dict param_dict,
    py::dict buffer_dict,
    int i_layer
) {
    std::string layer_prefix = "layer" + std::to_string(i_layer);

    // Retrieve parameters and buffers
    torch::Tensor reduction_weight = param_dict[py::str("reduction_weight_" + layer_prefix)].cast<torch::Tensor>();
    torch::Tensor norm_weight = param_dict[py::str("norm_weight_" + layer_prefix)].cast<torch::Tensor>();
    torch::Tensor norm_bias = param_dict[py::str("norm_bias_" + layer_prefix)].cast<torch::Tensor>();
    torch::Tensor input_resolution_tensor = buffer_dict[py::str("input_resolution_" + layer_prefix)].cast<torch::Tensor>();
    // auto input_resolution = input_resolution_tensor.accessor<long, 1>();
    // int H = input_resolution[0];
    // int W = input_resolution[1];
    auto H = input_resolution_tensor[0].item<int>();
    auto W = input_resolution_tensor[1].item<int>();

    // Implementation
    auto sizes = x.sizes();
    int64_t B = sizes[0];
    int64_t L = sizes[1];
    int64_t C = sizes[2];
    if (L != H * W) {
        throw std::runtime_error("input feature has wrong size");
    }
    if (H % 2 != 0 || W % 2 != 0) {
        throw std::runtime_error("x size (" + std::to_string(H) + "*" + std::to_string(W) + ") are not even.");
    }

    x = x.view({B, H, W, C});
    torch::Tensor x0 = x.slice(1, 0, c10::nullopt, 2).slice(2, 0, c10::nullopt, 2);
    torch::Tensor x1 = x.slice(1, 1, c10::nullopt, 2).slice(2, 0, c10::nullopt, 2);
    torch::Tensor x2 = x.slice(1, 0, c10::nullopt, 2).slice(2, 1, c10::nullopt, 2);
    torch::Tensor x3 = x.slice(1, 1, c10::nullopt, 2).slice(2, 1, c10::nullopt, 2);
    x = torch::cat({x0, x1, x2, x3}, -1);
    x = x.view({B, -1, 4 * C});

    x = torch::nn::functional::linear(x, reduction_weight);
    x = torch::layer_norm(x, {x.size(-1)}, norm_weight, norm_bias);

    return x;
}

// basic_layer_forward: Basic layer processing multiple blocks
torch::Tensor basic_layer_forward(
    torch::Tensor x,
    py::dict param_dict,
    py::dict buffer_dict,
    int i_layer
) {
    torch::Tensor depths = buffer_dict[py::str("depths")].cast<torch::Tensor>();
    int num_blocks = depths[i_layer].item<int>();

    for (int i_block = 0; i_block < num_blocks; ++i_block) {
        x = swin_transformer_block_forward(x, param_dict, buffer_dict, i_layer, i_block);
    }

    int num_layers = buffer_dict[py::str("num_layers")].cast<torch::Tensor>().item<int>();
    if (i_layer < num_layers - 1) {
        x = patch_merging_forward(x, param_dict, buffer_dict, i_layer);
    }

    return x;
}

// patch_embed_forward: Patch embedding forward pass
torch::Tensor patch_embed_forward(
    torch::Tensor x,
    py::dict param_dict,
    py::dict buffer_dict
) {
    torch::Tensor proj_weight = param_dict[py::str("patch_embed_proj")].cast<torch::Tensor>();
    torch::Tensor proj_bias = param_dict[py::str("patch_embed_bias")].cast<torch::Tensor>();
    torch::Tensor patch_size_tensor = buffer_dict[py::str("patch_size")].cast<torch::Tensor>();
    auto patch_height = patch_size_tensor[0].item<int>();
    auto patch_width = patch_size_tensor[1].item<int>();

    torch::nn::functional::Conv2dFuncOptions options;
    options.stride({patch_height, patch_width});
    options.bias(proj_bias);
    x = torch::nn::functional::conv2d(x, proj_weight, options);
    x = x.flatten(2).transpose(1, 2);

    if (param_dict.contains(py::str("patch_embed_norm_weight"))) {
        torch::Tensor norm_weight = param_dict[py::str("patch_embed_norm_weight")].cast<torch::Tensor>();
        torch::Tensor norm_bias = param_dict[py::str("patch_embed_norm_bias")].cast<torch::Tensor>();
        x = torch::layer_norm(x, {x.size(-1)}, norm_weight, norm_bias);
    }

    return x;
}

// model_forward: Full model forward pass
torch::Tensor model_forward(
    torch::Tensor x,
    py::dict param_dict,
    py::dict buffer_dict
) {
    x = patch_embed_forward(x, param_dict, buffer_dict);
    
    double drop_rate = buffer_dict[py::str("drop_rate")].cast<torch::Tensor>().item<double>();
    x = torch::nn::functional::dropout(
        x, torch::nn::functional::DropoutFuncOptions().p(drop_rate).training(true)
    );

    int num_layers = buffer_dict[py::str("num_layers")].cast<torch::Tensor>().item<int>();
    for (int i_layer = 0; i_layer < num_layers; ++i_layer) {
        x = basic_layer_forward(x, param_dict, buffer_dict, i_layer);
    }

    torch::Tensor norm_weight = param_dict[py::str("norm_weight")].cast<torch::Tensor>();
    torch::Tensor norm_bias = param_dict[py::str("norm_bias")].cast<torch::Tensor>();
    x = torch::layer_norm(x, {x.size(-1)}, norm_weight, norm_bias);

    x = x.transpose(1, 2);  // B C L
    x = torch::adaptive_avg_pool1d(x, 1);  // B C 1
    x = x.flatten(1);  // B C


    if (param_dict.contains(py::str("head_weight"))) {
        torch::Tensor head_weight = param_dict[py::str("head_weight")].cast<torch::Tensor>();
        torch::Tensor head_bias = param_dict[py::str("head_bias")].cast<torch::Tensor>();
        x = torch::nn::functional::linear(x, head_weight, head_bias);
    }

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &model_forward, "Swin Transformer model forward pass");
}
#include <torch/torch.h>

#include <torch/torch.h>

torch::Tensor window_partition(torch::Tensor x, std::int64_t window_size) {
    auto B = x.size(0);
    auto H = x.size(1);
    auto W = x.size(2);
    auto C = x.size(3);
    
    x = x.view({B, H / window_size, window_size, W / window_size, window_size, C});
    auto windows = x.permute({0, 1, 3, 2, 4, 5}).contiguous().view({-1, window_size, window_size, C});
    
    return windows;
}

torch::Tensor window_reverse(torch::Tensor windows, std::int64_t window_size, std::int64_t H, std::int64_t W) {
    auto B = std::int64_t(windows.size(0) / (H * W / window_size / window_size));
    
    auto x = windows.view({B, H / window_size, W / window_size, window_size, window_size, -1});
    x = x.permute({0, 1, 3, 2, 4, 5}).contiguous().view({B, H, W, -1});
    
    return x;
}

torch::Tensor mlp_fn(
    torch::Tensor x, 
    torch::Tensor fc1_weight, 
    torch::Tensor fc1_bias, 
    torch::Tensor fc2_weight, 
    torch::Tensor fc2_bias, 
    std::string act_layer = "gelu",
    float drop_rate = 0.0
) {
    x = torch::linear(x, fc1_weight, fc1_bias);
    
    if (act_layer == "gelu") {
        x = torch::gelu(x);
    } else if (act_layer == "relu") {
        x = torch::relu(x);
    }
    
    x = torch::dropout(x, drop_rate, true);
    x = torch::linear(x, fc2_weight, fc2_bias);
    x = torch::dropout(x, drop_rate, true);
    
    return x;
}

torch::Tensor swin_mlp_block_fn(
    torch::Tensor x, 
    torch::Tensor norm1_weight, 
    torch::Tensor norm1_bias, 
    torch::Tensor spatial_mlp_weight, 
    torch::Tensor spatial_mlp_bias,
    torch::Tensor norm2_weight, 
    torch::Tensor norm2_bias, 
    torch::Tensor fc1_weight, 
    torch::Tensor fc1_bias, 
    torch::Tensor fc2_weight, 
    torch::Tensor fc2_bias,
    std::vector<std::int64_t> input_resolution, 
    std::int64_t num_heads, 
    std::int64_t window_size, 
    std::int64_t shift_size, 
    float mlp_ratio,
    float drop_rate,
    float drop_path_rate,
    std::string act_layer = "gelu"
) {
    std::vector<std::int64_t> padding = {
        window_size - shift_size,
        shift_size,
        window_size - shift_size,
        shift_size
    };
    std::int64_t P_l = padding[0], P_r = padding[1], P_t = padding[2], P_b = padding[3];

    std::int64_t H = input_resolution[0];
    std::int64_t W = input_resolution[1];
    std::int64_t B = x.size(0);
    std::int64_t L = x.size(1);
    std::int64_t C = x.size(2);
    auto shortcut = x;

    // norm1
    x = torch::layer_norm(x, {C}, norm1_weight, norm1_bias);
    x = x.view({B, H, W, C});

    // shift
    torch::Tensor shifted_x;
    if (shift_size > 0) {
        shifted_x = torch::constant_pad_nd(x, {0, 0, P_l, P_r, P_t, P_b}, 0);
    } else {
        shifted_x = x;
    }
    std::int64_t _H = shifted_x.size(1);
    std::int64_t _W = shifted_x.size(2);

    // window partition
    auto x_windows = window_partition(shifted_x, window_size);
    x_windows = x_windows.view({-1, window_size * window_size, C});

    // spatial mlp
    auto x_windows_heads = x_windows.view({-1, window_size * window_size, num_heads, C / num_heads});
    x_windows_heads = x_windows_heads.transpose(1, 2);
    x_windows_heads = x_windows_heads.reshape({-1, num_heads * window_size * window_size, C / num_heads});

    auto spatial_mlp_windows = torch::conv1d(
        x_windows_heads, 
        spatial_mlp_weight, 
        spatial_mlp_bias, 
        /*stride=*/1,
        /*padding=*/torch::IntArrayRef(0),
        /*dilation=*/1,
        /*groups=*/num_heads
    );
    
    spatial_mlp_windows = spatial_mlp_windows.view({-1, num_heads, window_size * window_size, C / num_heads}).transpose(1, 2);
    spatial_mlp_windows = spatial_mlp_windows.reshape({-1, window_size * window_size, C});

    // merge windows
    spatial_mlp_windows = spatial_mlp_windows.reshape({-1, window_size, window_size, C});
    shifted_x = window_reverse(spatial_mlp_windows, window_size, _H, _W);

    // reverse shift
    torch::Tensor x_unshifted;
    if (shift_size > 0) {
        x_unshifted = shifted_x.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(P_t, _H - P_b),
            torch::indexing::Slice(P_l, _W - P_r),
            torch::indexing::Slice()
        }).contiguous();
    } else {
        x_unshifted = shifted_x;
    }

    x = x_unshifted.view({B, H * W, C});

    // FFN
    x = shortcut + x;
    x = x + mlp_fn(
        torch::layer_norm(x, {C}, norm2_weight, norm2_bias),
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        act_layer,
        drop_rate
    );
    
    return x;
}

torch::Tensor patch_merging_fn(
    torch::Tensor x, 
    torch::Tensor norm_weight, 
    torch::Tensor norm_bias, 
    torch::Tensor reduction_weight, 
    torch::Tensor reduction_bias, 
    std::vector<std::int64_t> input_resolution, 
    std::int64_t dim
) {
    std::int64_t H = input_resolution[0];
    std::int64_t W = input_resolution[1];
    std::int64_t B = x.size(0);
    std::int64_t L = x.size(1);
    std::int64_t C = x.size(2);
    
    x = x.view({B, H, W, C});

    auto x0 = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2), 
                       torch::indexing::Slice(0, torch::indexing::None, 2), torch::indexing::Slice()});
    auto x1 = x.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2), 
                       torch::indexing::Slice(0, torch::indexing::None, 2), torch::indexing::Slice()});
    auto x2 = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2), 
                       torch::indexing::Slice(1, torch::indexing::None, 2), torch::indexing::Slice()});
    auto x3 = x.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2), 
                       torch::indexing::Slice(1, torch::indexing::None, 2), torch::indexing::Slice()});
                       
    x = torch::cat({x0, x1, x2, x3}, -1);
    x = x.view({B, -1, 4 * C});

    x = torch::layer_norm(x, {4 * C}, norm_weight, norm_bias);
    x = torch::linear(x, reduction_weight, reduction_bias);
    
    return x;
}

torch::Tensor basic_layer_fn(
    torch::Tensor x,
    std::map<std::string, torch::Tensor> params,
    std::vector<std::int64_t> input_resolution,
    std::int64_t depth,
    std::int64_t num_heads,
    std::int64_t window_size,
    float mlp_ratio,
    float drop_rate,
    float drop_path_rate,
    bool downsample,
    std::int64_t i_layer
) {
    for (std::int64_t i = 0; i < depth; i++) {
        x = swin_mlp_block_fn(
            x,
            params["layer" + std::to_string(i_layer) + "_norm1_weight_" + std::to_string(i)],
            params["layer" + std::to_string(i_layer) + "_norm1_bias_" + std::to_string(i)],
            params["layer" + std::to_string(i_layer) + "_spatial_mlp_weight_" + std::to_string(i)],
            params["layer" + std::to_string(i_layer) + "_spatial_mlp_bias_" + std::to_string(i)],
            params["layer" + std::to_string(i_layer) + "_norm2_weight_" + std::to_string(i)],
            params["layer" + std::to_string(i_layer) + "_norm2_bias_" + std::to_string(i)],
            params["layer" + std::to_string(i_layer) + "_fc1_weight_" + std::to_string(i)],
            params["layer" + std::to_string(i_layer) + "_fc1_bias_" + std::to_string(i)],
            params["layer" + std::to_string(i_layer) + "_fc2_weight_" + std::to_string(i)],
            params["layer" + std::to_string(i_layer) + "_fc2_bias_" + std::to_string(i)],
            input_resolution,
            num_heads,
            window_size,
            (i % 2 == 0) ? 0 : window_size / 2,
            mlp_ratio,
            drop_rate,
            drop_path_rate
        );
    }
    
    if (downsample) {
        x = patch_merging_fn(
            x,
            params["layer" + std::to_string(i_layer) + "_downsample_norm_weight"],
            params["layer" + std::to_string(i_layer) + "_downsample_norm_bias"],
            params["layer" + std::to_string(i_layer) + "_downsample_reduction_weight"],
            params["layer" + std::to_string(i_layer) + "_downsample_reduction_bias"],
            input_resolution,
            params["layer" + std::to_string(i_layer) + "_dim"].item<std::int64_t>()
        );
    }
    
    return x;
}

torch::Tensor patch_embed_fn(
    torch::Tensor x,
    torch::Tensor proj_weight,
    torch::Tensor proj_bias,
    torch::Tensor norm_weight = torch::Tensor(),
    torch::Tensor norm_bias = torch::Tensor(),
    std::vector<std::int64_t> img_size = {224, 224},
    std::vector<std::int64_t> patch_size = {4, 4}
) {
    std::int64_t B = x.size(0);
    std::int64_t C = x.size(1);
    std::int64_t H = x.size(2);
    std::int64_t W = x.size(3);
    
    x = torch::conv2d(x, proj_weight, proj_bias, 
        /*stride=*/{4, 4});
    x = x.flatten(2).transpose(1, 2);
    
    if (norm_weight.defined()) {
        x = torch::layer_norm(x, {x.size(-1)}, norm_weight, norm_bias);
    }
    
    return x;
}

torch::Tensor model_fn(
    torch::Tensor x, 
    torch::Tensor proj_weight, 
    torch::Tensor proj_bias,
    torch::Tensor norm_weight, 
    torch::Tensor norm_bias,
    std::vector<std::int64_t> patches_resolution, 
    std::int64_t patch_size, 
    std::int64_t embed_dim,
    std::int64_t num_layers, 
    std::map<std::string, torch::Tensor> layer_params, 
    std::vector<std::int64_t> depths,
    std::vector<std::int64_t> num_heads, 
    std::int64_t window_size, 
    float mlp_ratio,
    float drop_rate, 
    float drop_path_rate, 
    std::int64_t num_features,
    torch::Tensor norm_weight_2, 
    torch::Tensor norm_bias_2,
    torch::Tensor head_weight, 
    torch::Tensor head_bias, 
    bool training
) {
    // Patch embed
    x = patch_embed_fn(
        x, 
        proj_weight, 
        proj_bias, 
        norm_weight, 
        norm_bias, 
        patches_resolution, 
        {patch_size, patch_size}
    );
    
    x = torch::dropout(x, drop_rate, training);

    // Layers
    for (std::int64_t i_layer = 0; i_layer < num_layers; i_layer++) {
        std::int64_t dim = std::int64_t(embed_dim * std::pow(2, i_layer));
        std::vector<std::int64_t> input_resolution = {
            patches_resolution[0] / std::int64_t(std::pow(2, i_layer)),
            patches_resolution[1] / std::int64_t(std::pow(2, i_layer))
        };
        
        x = basic_layer_fn(
            x,
            layer_params,
            input_resolution,
            depths[i_layer],
            num_heads[i_layer],
            window_size,
            mlp_ratio,
            drop_rate,
            drop_path_rate,
            i_layer < num_layers - 1,
            i_layer
        );
    }

    // Final norm and head
    x = torch::layer_norm(x, {num_features}, norm_weight_2, norm_bias_2);
    x = x.transpose(1, 2);
    x = torch::adaptive_avg_pool1d(x, 1).squeeze(-1);
    x = torch::linear(x, head_weight, head_bias);
    
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &model_fn, "SwinMLP");
}
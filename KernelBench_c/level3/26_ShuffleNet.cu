#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>

torch::Tensor channel_shuffle(torch::Tensor x, int groups) {
    int batch_size = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int channels_per_group = channels / groups;
    x = x.view({batch_size, groups, channels_per_group, height, width});
    x = x.transpose(1, 2).contiguous();
    x = x.view({batch_size, channels, height, width});
    return x;
}

torch::Tensor shuffle_net_unit(
    torch::Tensor x,
    torch::Tensor conv1_weight, torch::Tensor bn1_weight, torch::Tensor bn1_bias,
    torch::Tensor bn1_running_mean, torch::Tensor bn1_running_var,
    torch::Tensor conv2_weight, torch::Tensor bn2_weight, torch::Tensor bn2_bias,
    torch::Tensor bn2_running_mean, torch::Tensor bn2_running_var,
    torch::Tensor conv3_weight, torch::Tensor bn3_weight, torch::Tensor bn3_bias,
    torch::Tensor bn3_running_mean, torch::Tensor bn3_running_var,
    torch::Tensor shortcut_conv_weight, torch::Tensor shortcut_bn_weight, torch::Tensor shortcut_bn_bias,
    torch::Tensor shortcut_bn_running_mean, torch::Tensor shortcut_bn_running_var,
    int in_channels, int out_channels, int groups) {
    
    std::cout << "Input (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << x[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    torch::Tensor out = torch::conv2d(x, conv1_weight, c10::nullopt, torch::IntArrayRef({1, 1}), torch::IntArrayRef({0, 0}), torch::IntArrayRef({1, 1}), groups);
    std::cout << "After conv1 (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << out[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    out = torch::batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, false, 0, 1e-5, true);
    out = torch::relu(out);
    std::cout << "After bn1+relu (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << out[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    out = torch::conv2d(out, conv2_weight, c10::nullopt, torch::IntArrayRef({1, 1}), torch::IntArrayRef({1, 1}), torch::IntArrayRef({1, 1}), out.size(1));
    std::cout << "After conv2 (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << out[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    out = torch::batch_norm(out, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, false, 0, 1e-5, true);
    std::cout << "After bn2 (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << out[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    out = channel_shuffle(out, groups);
    std::cout << "After channel_shuffle (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << out[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    out = torch::conv2d(out, conv3_weight, c10::nullopt, torch::IntArrayRef({1, 1}), torch::IntArrayRef({0, 0}), torch::IntArrayRef({1, 1}), groups);
    std::cout << "After conv3 (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << out[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    out = torch::batch_norm(out, bn3_weight, bn3_bias, bn3_running_mean, bn3_running_var, false, 0, 1e-5, true);
    out = torch::relu(out);
    std::cout << "After bn3+relu (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << out[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    torch::Tensor shortcut;
    if (in_channels == out_channels) {
        shortcut = x;
    } else {
        std::cout << "Shortcut conv weight (first 5): ";
        auto flat_weights = shortcut_conv_weight.flatten();
        for (int j = 0; j < std::min(5L, flat_weights.size(0)); ++j) {
            std::cout << flat_weights[j].item<float>() << " ";
        }
        std::cout << std::endl;

        shortcut = torch::conv2d(x, shortcut_conv_weight, c10::nullopt, torch::IntArrayRef({1, 1}), torch::IntArrayRef({0, 0}), torch::IntArrayRef({1, 1}), 1);
        std::cout << "Shortcut after conv (first 5): ";
        for (int j = 0; j < 5; ++j) std::cout << shortcut[0][0][0][j].item<float>() << " ";
        std::cout << std::endl;
        if (shortcut_bn_running_mean.defined() && shortcut_bn_running_var.defined()) {
            shortcut = torch::batch_norm(shortcut, shortcut_bn_weight, shortcut_bn_bias, 
                                         shortcut_bn_running_mean, shortcut_bn_running_var, false, 0, 1e-5, true);
            std::cout << "Shortcut after bn (first 5): ";
            for (int j = 0; j < 5; ++j) std::cout << shortcut[0][0][0][j].item<float>() << " ";
            std::cout << std::endl;
        }
    }
    out = out + shortcut;
    return out;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv1_weight, torch::Tensor bn1_weight, torch::Tensor bn1_bias,
    torch::Tensor bn1_running_mean, torch::Tensor bn1_running_var,
    torch::Tensor conv5_weight, torch::Tensor bn5_weight, torch::Tensor bn5_bias,
    torch::Tensor bn5_running_mean, torch::Tensor bn5_running_var,
    torch::Tensor fc_weight, torch::Tensor fc_bias,
    const std::vector<torch::Tensor>& stage2_params,
    const std::vector<torch::Tensor>& stage3_params,
    const std::vector<torch::Tensor>& stage4_params,
    int in_channels, int groups, int stage2_repeats, int stage3_repeats, int stage4_repeats,
    const std::vector<int64_t>& stages_out_channels) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");

    // conv1
    x = torch::conv2d(x, conv1_weight, c10::nullopt, torch::IntArrayRef({2, 2}), torch::IntArrayRef({1, 1}), torch::IntArrayRef({1, 1}), 1);
    cudaDeviceSynchronize();
    std::cout << "CUDA conv1 output (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    x = torch::batch_norm(x, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, false, 0, 1e-5, true);
    x = torch::relu(x);
    cudaDeviceSynchronize();
    std::cout << "CUDA bn1 output (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    x = torch::max_pool2d(x, {3, 3}, {2, 2}, {1, 1});
    cudaDeviceSynchronize();
    std::cout << "CUDA maxpool output (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    // stage2
    for (int i = 0; i < stage2_repeats; ++i) {
        std::cout << "Stage2 unit " << i << " input (first 5): ";
        for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
        std::cout << std::endl;

        int base_idx = i * 20;
        int unit_in_channels = (i == 0) ? stages_out_channels[0] : stages_out_channels[1];
        int out_channels = stages_out_channels[1];

        x = shuffle_net_unit(
            x, stage2_params[base_idx], stage2_params[base_idx + 1], stage2_params[base_idx + 2],
            stage2_params[base_idx + 3], stage2_params[base_idx + 4],
            stage2_params[base_idx + 5], stage2_params[base_idx + 6], stage2_params[base_idx + 7],
            stage2_params[base_idx + 8], stage2_params[base_idx + 9],
            stage2_params[base_idx + 10], stage2_params[base_idx + 11], stage2_params[base_idx + 12],
            stage2_params[base_idx + 13], stage2_params[base_idx + 14],
            stage2_params[base_idx + 15], stage2_params[base_idx + 16], stage2_params[base_idx + 17],
            stage2_params[base_idx + 18], stage2_params[base_idx + 19],
            unit_in_channels, out_channels, groups
        );

        cudaDeviceSynchronize();
        std::cout << "CUDA stage2 unit " << i << " output (first 5): ";
        for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
        std::cout << std::endl;
    }

    // stage3
    for (int i = 0; i < stage3_repeats; ++i) {
        std::cout << "Stage3 unit " << i << " input (first 5): ";
        for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
        std::cout << std::endl;

        int base_idx = i * 20;
        int unit_in_channels = (i == 0) ? stages_out_channels[1] : stages_out_channels[2];
        int out_channels = stages_out_channels[2];

        x = shuffle_net_unit(
            x, stage3_params[base_idx], stage3_params[base_idx + 1], stage3_params[base_idx + 2],
            stage3_params[base_idx + 3], stage3_params[base_idx + 4],
            stage3_params[base_idx + 5], stage3_params[base_idx + 6], stage3_params[base_idx + 7],
            stage3_params[base_idx + 8], stage3_params[base_idx + 9],
            stage3_params[base_idx + 10], stage3_params[base_idx + 11], stage3_params[base_idx + 12],
            stage3_params[base_idx + 13], stage3_params[base_idx + 14],
            stage3_params[base_idx + 15], stage3_params[base_idx + 16], stage3_params[base_idx + 17],
            stage3_params[base_idx + 18], stage3_params[base_idx + 19],
            unit_in_channels, out_channels, groups
        );

        cudaDeviceSynchronize();
        std::cout << "CUDA stage3 unit " << i << " output (first 5): ";
        for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
        std::cout << std::endl;
    }

    // stage4
    for (int i = 0; i < stage4_repeats; ++i) {
        std::cout << "Stage4 unit " << i << " input (first 5): ";
        for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
        std::cout << std::endl;

        int base_idx = i * 20;
        int unit_in_channels = (i == 0) ? stages_out_channels[2] : stages_out_channels[3];
        int out_channels = stages_out_channels[3];

        x = shuffle_net_unit(
            x, stage4_params[base_idx], stage4_params[base_idx + 1], stage4_params[base_idx + 2],
            stage4_params[base_idx + 3], stage4_params[base_idx + 4],
            stage4_params[base_idx + 5], stage4_params[base_idx + 6], stage4_params[base_idx + 7],
            stage4_params[base_idx + 8], stage4_params[base_idx + 9],
            stage4_params[base_idx + 10], stage4_params[base_idx + 11], stage4_params[base_idx + 12],
            stage4_params[base_idx + 13], stage4_params[base_idx + 14],
            stage4_params[base_idx + 15], stage4_params[base_idx + 16], stage4_params[base_idx + 17],
            stage4_params[base_idx + 18], stage4_params[base_idx + 19],
            unit_in_channels, out_channels, groups
        );

        cudaDeviceSynchronize();
        std::cout << "CUDA stage4 unit " << i << " output (first 5): ";
        for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
        std::cout << std::endl;
    }

    // conv5
    x = torch::conv2d(x, conv5_weight, c10::nullopt, torch::IntArrayRef({1, 1}), torch::IntArrayRef({0, 0}), torch::IntArrayRef({1, 1}), 1);
    cudaDeviceSynchronize();
    std::cout << "CUDA conv5 output (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    x = torch::batch_norm(x, bn5_weight, bn5_bias, bn5_running_mean, bn5_running_var, false, 0, 1e-5, true);
    x = torch::relu(x);
    cudaDeviceSynchronize();
    std::cout << "CUDA bn5 output (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][0][0][j].item<float>() << " ";
    std::cout << std::endl;

    // Global average pooling
    x = torch::avg_pool2d(x, {x.size(2), x.size(3)});
    cudaDeviceSynchronize();
    std::cout << "CUDA avgpool output (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][j].item<float>() << " ";
    std::cout << std::endl;

    // Fully connected layer
    x = x.view({x.size(0), -1});
    x = torch::linear(x, fc_weight, fc_bias);
    cudaDeviceSynchronize();
    std::cout << "CUDA fc output (first 5): ";
    for (int j = 0; j < 5; ++j) std::cout << std::fixed << std::setprecision(6) << x[0][j].item<float>() << " ";
    std::cout << std::endl;

    return x;
}


// Explicitly define the binding with all parameters
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ShuffleNet Forward (CUDA)",
          py::arg("x"),
          py::arg("conv1_weight"), py::arg("bn1_weight"), py::arg("bn1_bias"),
          py::arg("bn1_running_mean"), py::arg("bn1_running_var"),
          py::arg("conv5_weight"), py::arg("bn5_weight"), py::arg("bn5_bias"),
          py::arg("bn5_running_mean"), py::arg("bn5_running_var"),
          py::arg("fc_weight"), py::arg("fc_bias"),
          py::arg("stage2_params"), py::arg("stage3_params"), py::arg("stage4_params"),
          py::arg("in_channels"), py::arg("groups"),
          py::arg("stage2_repeats"), py::arg("stage3_repeats"), py::arg("stage4_repeats"),
          py::arg("stages_out_channels"));
}
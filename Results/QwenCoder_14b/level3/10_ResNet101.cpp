#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <omp.h>

namespace py = pybind11;

// CPU version of the conv2d operation
template <typename scalar_t>
void conv2d_cpu(const scalar_t* input, const scalar_t* kernel, scalar_t* output,
               int width, int height, int ksize, int stride, int padding) {
    #pragma omp parallel for collapse(2)
    for (int tidy = 0; tidy < height; ++tidy) {
        for (int tidx = 0; tidx < width; ++tidx) {
            int kernel_radius = ksize / 2;
            scalar_t sum = 0;
            for (int i = -kernel_radius; i <= kernel_radius; ++i) {
                for (int j = -kernel_radius; j <= kernel_radius; ++j) {
                    int x = tidx * stride + j - padding;
                    int y = tidy * stride + i - padding;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        sum += input[y * width + x] * kernel[(i + kernel_radius) * ksize + (j + kernel_radius)];
                    }
                }
            }
            output[tidy * width + tidx] = sum;
        }
    }
}

// CPU version of the bottleneck function
torch::Tensor bottleneck_fn_cpu(
    torch::Tensor x,
    const torch::Tensor& conv1_w,
    const torch::Tensor& conv2_w,
    const torch::Tensor& conv3_w,
    const torch::Tensor& bn1_w,
    const torch::Tensor& bn1_b,
    const torch::Tensor& bn1_m,
    const torch::Tensor& bn1_v,
    const torch::Tensor& bn2_w,
    const torch::Tensor& bn2_b,
    const torch::Tensor& bn2_m,
    const torch::Tensor& bn2_v,
    const torch::Tensor& bn3_w,
    const torch::Tensor& bn3_b,
    const torch::Tensor& bn3_m,
    const torch::Tensor& bn3_v,
    const torch::Tensor& downsample_conv_w,
    const torch::Tensor& downsample_bn_w,
    const torch::Tensor& downsample_bn_b,
    const torch::Tensor& downsample_bn_m,
    const torch::Tensor& downsample_bn_v,
    int64_t stride,
    bool is_training
) {
    torch::Tensor identity = x;
    bool has_downsample = downsample_conv_w.defined();

    torch::Tensor downsample_out;
    if (has_downsample) {
        downsample_out = torch::conv2d(x, downsample_conv_w, /*bias=*/torch::Tensor(), stride);
        downsample_out = torch::batch_norm(downsample_out, downsample_bn_w, downsample_bn_b, 
            downsample_bn_m, downsample_bn_v, is_training, 0.1, 1e-5, true);
    }

    torch::Tensor out = torch::conv2d(x, conv1_w, /*bias=*/torch::Tensor());
    out = torch::batch_norm(out, bn1_w, bn1_b, bn1_m, bn1_v, is_training, 0.1, 1e-5, true);
    out = torch::relu(out);

    out = torch::conv2d(out, conv2_w, /*bias=*/torch::Tensor(), stride, /*padding=*/1);
    out = torch::batch_norm(out, bn2_w, bn2_b, bn2_m, bn2_v, is_training, 0.1, 1e-5, true);
    out = torch::relu(out);

    out = torch::conv2d(out, conv3_w, /*bias=*/torch::Tensor());
    out = torch::batch_norm(out, bn3_w, bn3_b, bn3_m, bn3_v, is_training, 0.1, 1e-5, true);

    identity = has_downsample ? downsample_out : identity.to(out.dtype());
    out = out + identity;
    return torch::relu(out);
}

torch::Tensor forward_cpu(
    torch::Tensor x,
    py::object params,
    bool is_training
) {
    // Pre-fetch all parameters in contiguous memory blocks
    auto device = x.device();
    std::vector<torch::Tensor> param_buffers;

    // Initial layer parameters
    std::vector<torch::Tensor> initial_params{
        params.attr("get")("conv1_w").cast<torch::Tensor>(),
        params.attr("get")("bn1_w").cast<torch::Tensor>(),
        params.attr("get")("bn1_b").cast<torch::Tensor>(),
        params.attr("get")("bn1_m").cast<torch::Tensor>(),
        params.attr("get")("bn1_v").cast<torch::Tensor>()
    };
    for (auto& p : initial_params) p = p.contiguous().to(device, /*non_blocking=*/true);

    x = torch::conv2d(x, initial_params[0], /*bias=*/torch::Tensor(), 2, 3);
    x = torch::batch_norm(x, initial_params[1], initial_params[2], initial_params[3], initial_params[4], 
                        is_training, 0.1, 1e-5, true);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    // Layer processing with batched parameter transfers
    for (int layer_idx = 1; layer_idx <= 4; ++layer_idx) {
        std::string key = "layer" + std::to_string(layer_idx) + "_blocks";
        py::list blocks = params.attr("get")(py::str(key)).cast<py::list>();

        // Pre-fetch all block parameters
        std::vector<std::vector<torch::Tensor>> layer_params;
        for (auto block : blocks) {
            py::object bp = block.cast<py::object>();
            std::vector<torch::Tensor> block_tensors;
            
            const char* names[] = {"conv1_w", "conv2_w", "conv3_w",
                                  "bn1_w", "bn1_b", "bn1_m", "bn1_v",
                                  "bn2_w", "bn2_b", "bn2_m", "bn2_v",
                                  "bn3_w", "bn3_b", "bn3_m", "bn3_v"};
            
            for (const char* name : names) {
                block_tensors.push_back(bp.attr("get")(py::str(name)).cast<torch::Tensor>());
            }

            if (py::bool_(bp.attr("__contains__")("downsample_conv_w"))) {
                const char* ds_names[] = {"downsample_conv_w", "downsample_bn_w",
                                         "downsample_bn_b", "downsample_bn_m", "downsample_bn_v"};
                for (const char* ds_name : ds_names) {
                    block_tensors.push_back(bp.attr("get")(py::str(ds_name)).cast<torch::Tensor>());
                }
            }
            
            layer_params.push_back(block_tensors);
        }

        // Batch transfer for layer
        for (auto& block_tensors : layer_params) {
            for (auto& t : block_tensors) {
                t = t.contiguous().to(device, /*non_blocking=*/true);
            }
        }

        // Process blocks with pre-fetched parameters
        for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
            auto& block_tensors = layer_params[block_idx];
            int64_t stride = (block_idx == 0 && layer_idx > 1) ? 2 : 1;
            bool has_downsample = block_tensors.size() > 15;

            x = bottleneck_fn_cpu(x,
                block_tensors[0], block_tensors[1], block_tensors[2],
                block_tensors[3], block_tensors[4], block_tensors[5], block_tensors[6],
                block_tensors[7], block_tensors[8], block_tensors[9], block_tensors[10],
                block_tensors[11], block_tensors[12], block_tensors[13], block_tensors[14],
                has_downsample ? block_tensors[15] : torch::Tensor(),
                has_downsample ? block_tensors[16] : torch::Tensor(),
                has_downsample ? block_tensors[17] : torch::Tensor(),
                has_downsample ? block_tensors[18] : torch::Tensor(),
                has_downsample ? block_tensors[19] : torch::Tensor(),
                stride, is_training
            );
        }
    }

    x = torch::adaptive_avg_pool2d(x, {1, 1}).contiguous();
    x = x.view({x.size(0), -1});

    auto fc_w = params.attr("get")("fc_w").cast<torch::Tensor>().contiguous().to(device);
    auto fc_b = params.attr("get")("fc_b").cast<torch::Tensor>().contiguous().to(device);
    return torch::linear(x, fc_w, fc_b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "ResNet101 forward CPU");
}
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <omp.h>

namespace py = pybind11;

void adaptive_avg_pool2d_cpu(const float* input, float* output, int N, int C, int H, int W) {
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    const int offset = ((n * C + c) * H + h) * W + w;
                    sum += input[offset];
                }
            }
            output[n * C + c] = sum / static_cast<float>(H * W);
        }
    }
}

void relu_cpu(float* data, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        data[i] = (data[i] > 0.f) ? data[i] : 0.f;
    }
}

torch::Tensor custom_relu(torch::Tensor input) {
    const size_t n = input.numel();
    relu_cpu(input.data_ptr<float>(), n);
    return input;
}

torch::Tensor forward(torch::Tensor x, py::object params_obj) {
    using namespace torch;
    
    std::map<std::string, Tensor> params;
    py::dict params_dict = params_obj.attr("items")();
    for (auto item : params_dict) {
        std::string key = py::cast<std::string>(item.first);
        Tensor value = py::cast<Tensor>(item.second);
        params[key] = value.contiguous();
    }

    if (!x.is_contiguous()) x = x.contiguous();
    
    x = conv2d(x, params["conv1_weight"], params["conv1_bias"],
               /*stride=*/at::IntArrayRef{2, 2},
               /*padding=*/at::IntArrayRef{0, 0},
               /*dilation=*/at::IntArrayRef{1, 1},
               /*groups=*/1);
    x = custom_relu(x);
    x = max_pool2d(x,
                   /*kernel_size=*/at::IntArrayRef{3, 3},
                   /*stride=*/at::IntArrayRef{2, 2},
                   /*padding=*/at::IntArrayRef{0, 0},
                   /*dilation=*/at::IntArrayRef{1, 1},
                   /*ceil_mode=*/true);

    auto fire_module = [&params](Tensor x, std::string prefix) {
        x = x.contiguous();
        Tensor squeeze = conv2d(x,
                              params[prefix + "_squeeze_weight"],
                              params[prefix + "_squeeze_bias"],
                              /*stride=*/at::IntArrayRef{1, 1},
                              /*padding=*/at::IntArrayRef{0, 0},
                              /*dilation=*/at::IntArrayRef{1, 1},
                              /*groups=*/1);
        squeeze = custom_relu(squeeze);

        squeeze = squeeze.contiguous();
        Tensor e1 = conv2d(squeeze,
                         params[prefix + "_expand1x1_weight"],
                         params[prefix + "_expand1x1_bias"],
                         /*stride=*/at::IntArrayRef{1, 1},
                         /*padding=*/at::IntArrayRef{0, 0},
                         /*dilation=*/at::IntArrayRef{1, 1},
                         /*groups=*/1);
        e1 = custom_relu(e1);
        
        Tensor e3 = conv2d(squeeze,
                         params[prefix + "_expand3x3_weight"],
                         params[prefix + "_expand3x3_bias"],
                         /*stride=*/at::IntArrayRef{1, 1},
                         /*padding=*/at::IntArrayRef{1, 1},
                         /*dilation=*/at::IntArrayRef{1, 1},
                         /*groups=*/1);
        e3 = custom_relu(e3);

        return cat({e1.contiguous(), e3.contiguous()}, /*dim=*/1);
    };

    x = fire_module(x, "fire1");
    x = fire_module(x, "fire2");
    x = fire_module(x, "fire3");
    x = max_pool2d(x,
                   /*kernel_size=*/at::IntArrayRef{3, 3},
                   /*stride=*/at::IntArrayRef{2, 2},
                   /*padding=*/at::IntArrayRef{0, 0},
                   /*dilation=*/at::IntArrayRef{1, 1},
                   /*ceil_mode=*/true);
    
    x = fire_module(x, "fire4");
    x = fire_module(x, "fire5");
    x = fire_module(x, "fire6");
    x = fire_module(x, "fire7");
    x = max_pool2d(x,
                   /*kernel_size=*/at::IntArrayRef{3, 3},
                   /*stride=*/at::IntArrayRef{2, 2},
                   /*padding=*/at::IntArrayRef{0, 0},
                   /*dilation=*/at::IntArrayRef{1, 1},
                   /*ceil_mode=*/true);
    
    x = fire_module(x, "fire8");

    x = x.contiguous();
    x = conv2d(x,
               params["classifier_weight"],
               params["classifier_bias"],
               /*stride=*/at::IntArrayRef{1, 1},
               /*padding=*/at::IntArrayRef{0, 0},
               /*dilation=*/at::IntArrayRef{1, 1},
               /*groups=*/1);
    x = custom_relu(x);
    
    auto sizes = x.sizes();
    auto out = at::empty({sizes[0], sizes[1], 1, 1}, x.options());
    
    adaptive_avg_pool2d_cpu(x.data_ptr<float>(), out.data_ptr<float>(), sizes[0], sizes[1], sizes[2], sizes[3]);
    
    return flatten(out, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SqueezeNet forward with CPU implementation");
}
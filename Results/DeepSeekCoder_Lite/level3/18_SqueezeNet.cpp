#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <omp.h>

namespace py = pybind11;

template<int BLOCK_SIZE = 256>
void adaptive_avg_pool2d_shared_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int N, int C, int H, int W) {
    int total = H * W;
    #pragma omp parallel for
    for (int idx = 0; idx < N * C; ++idx) {
        int n = idx / C;
        int c = idx % C;
        float thread_sum = 0.0f;
        #pragma omp simd
        for (int i = 0; i < total; ++i) {
            int h = i / W;
            int w = i % W;
            int offset = ((n * C + c) * H + h) * W + w;
            thread_sum += input[offset];
        }
        output[idx] = thread_sum / static_cast<float>(total);
    }
}

void relu_kernel(float* data, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        const float val = data[i];
        data[i] = (val > 0.f) ? val : 0.f;
    }
}

torch::Tensor custom_relu(torch::Tensor input) {
    const int threads = 256;
    const int n = input.numel();
    const int blocks = (n + threads - 1) / threads;
    relu_kernel(input.data_ptr<float>(), n);
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
    
    x = at::conv2d(x, params["conv1_weight"], params["conv1_bias"],
                   at::IntArrayRef({2, 2}), at::IntArrayRef({0, 0}),
                   at::IntArrayRef({1, 1}), 1);
    x = custom_relu(x);
    x = at::max_pool2d(x, at::IntArrayRef({3, 3}), at::IntArrayRef({2, 2}),
                        at::IntArrayRef({0, 0}), at::IntArrayRef({1, 1}), true);

    auto fire_module = [&params](Tensor x, std::string prefix) {
        x = x.contiguous();
        Tensor squeeze = at::conv2d(x,
                                   params[prefix + "_squeeze_weight"],
                                   params[prefix + "_squeeze_bias"],
                                   at::IntArrayRef({1, 1}), at::IntArrayRef({0, 0}),
                                   at::IntArrayRef({1, 1}), 1);
        squeeze = custom_relu(squeeze);

        squeeze = squeeze.contiguous();
        Tensor e1 = at::conv2d(squeeze,
                              params[prefix + "_expand1x1_weight"],
                              params[prefix + "_expand1x1_bias"],
                              at::IntArrayRef({1, 1}), at::IntArrayRef({0, 0}),
                              at::IntArrayRef({1, 1}), 1);
        e1 = custom_relu(e1);
        
        Tensor e3 = at::conv2d(squeeze,
                              params[prefix + "_expand3x3_weight"],
                              params[prefix + "_expand3x3_bias"],
                              at::IntArrayRef({1, 1}), at::IntArrayRef({1, 1}),
                              at::IntArrayRef({1, 1}), 1);
        e3 = custom_relu(e3);

        return at::cat({e1.contiguous(), e3.contiguous()}, 1);
    };

    x = fire_module(x, "fire1");
    x = fire_module(x, "fire2");
    x = fire_module(x, "fire3");
    x = at::max_pool2d(x, at::IntArrayRef({3, 3}), at::IntArrayRef({2, 2}),
                        at::IntArrayRef({0, 0}), at::IntArrayRef({1, 1}), true);
    
    x = fire_module(x, "fire4");
    x = fire_module(x, "fire5");
    x = fire_module(x, "fire6");
    x = fire_module(x, "fire7");
    x = at::max_pool2d(x, at::IntArrayRef({3, 3}), at::IntArrayRef({2, 2}),
                        at::IntArrayRef({0, 0}), at::IntArrayRef({1, 1}), true);
    
    x = fire_module(x, "fire8");

    x = x.contiguous();
    x = at::conv2d(x, params["classifier_weight"], params["classifier_bias"],
                   at::IntArrayRef({1, 1}), at::IntArrayRef({0, 0}),
                   at::IntArrayRef({1, 1}), 1);
    x = custom_relu(x);
    
    auto sizes = x.sizes();
    auto out = at::empty({sizes[0], sizes[1], 1, 1}, x.options());
    
    const int pool_blocks = sizes[0] * sizes[1];  // N * C
    const int pool_threads = 256;
    adaptive_avg_pool2d_shared_kernel<256>(input.data_ptr<float>(), output.data_ptr<float>(),
                                           sizes[0], sizes[1], sizes[2], sizes[3]);
    
    return at::flatten(out, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SqueezeNet forward with shared memory reduction");
}
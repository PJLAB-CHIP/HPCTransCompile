#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <omp.h>

namespace py = pybind11;

inline std::vector<int64_t> parseIntArrayRef(const py::object& obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        result.push_back(obj.cast<int64_t>());
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            result.push_back(py::cast<int64_t>(item));
        }
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    return result;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);

    // Get dimensions
    int64_t batch_size = x.size(0);
    int64_t in_channels = x.size(1);
    int64_t out_channels = weight.size(0);
    int64_t kernel_height = weight.size(2);
    int64_t kernel_width = weight.size(3);
    int64_t stride_height = stride_vec[0];
    int64_t stride_width = stride_vec[1];
    int64_t padding_height = padding_vec[0];
    int64_t padding_width = padding_vec[1];
    int64_t output_padding_height = output_padding_vec[0];
    int64_t output_padding_width = output_padding_vec[1];

    // Calculate output dimensions
    int64_t output_height = (x.size(2) - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;
    int64_t output_width = (x.size(3) - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;

    // Initialize output tensor
    torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    // Perform convolution transpose
    #pragma omp parallel for collapse(4)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t g = 0; g < groups; ++g) {
            for (int64_t oc = g * (out_channels / groups); oc < (g + 1) * (out_channels / groups); ++oc) {
                for (int64_t ic = g * (in_channels / groups); ic < (g + 1) * (in_channels / groups); ++ic) {
                    for (int64_t oh = 0; oh < output_height; ++oh) {
                        for (int64_t ow = 0; ow < output_width; ++ow) {
                            int64_t ih = (oh - output_padding_height + stride_height * kernel_height - 1) / stride_height;
                            int64_t iw = (ow - output_padding_width + stride_width * kernel_width - 1) / stride_width;
                            if (ih >= 0 && ih < x.size(2) && iw >= 0 && iw < x.size(3)) {
                                output[b][oc][oh][ow] += x[b][ic][ih][iw] * weight[oc][ic][oh % kernel_height][ow % kernel_width];
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias if provided
    if (bias.has_value()) {
        output += bias.value();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}

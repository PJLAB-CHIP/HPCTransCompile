#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Function to perform convolution
torch::Tensor conv2d_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto input_size = input.size();
    auto weight_size = weight.size();
    auto output_size = {input_size[0], weight_size[0], input_size[2] - weight_size[2] + 1, input_size[3] - weight_size[3] + 1};
    torch::Tensor output = torch::zeros(output_size);

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < output_size[0]; ++n) {
        for (int c_out = 0; c_out < output_size[1]; ++c_out) {
            for (int h_out = 0; h_out < output_size[2]; ++h_out) {
                for (int w_out = 0; w_out < output_size[3]; ++w_out) {
                    float sum = 0.0f;
                    for (int c_in = 0; c_in < weight_size[1]; ++c_in) {
                        for (int kh = 0; kh < weight_size[2]; ++kh) {
                            for (int kw = 0; kw < weight_size[3]; ++kw) {
                                sum += input[n][c_in][h_out + kh][w_out + kw].item<float>() * weight[c_out][c_in][kh][kw].item<float>();
                            }
                        }
                    }
                    output[n][c_out][h_out][w_out] = sum + bias[c_out].item<float>();
                }
            }
        }
    }

    return output;
}

// Function to perform batch normalization
torch::Tensor batch_norm_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, double momentum, double eps) {
    auto input_size = input.size();
    torch::Tensor output = torch::zeros(input_size);

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < input_size[0]; ++n) {
        for (int c = 0; c < input_size[1]; ++c) {
            for (int h = 0; h < input_size[2]; ++h) {
                for (int w = 0; w < input_size[3]; ++w) {
                    float x = input[n][c][h][w].item<float>();
                    float mean = running_mean[c].item<float>();
                    float var = running_var[c].item<float>();
                    float norm = (x - mean) / std::sqrt(var + eps);
                    output[n][c][h][w] = weight[c].item<float>() * norm + bias[c].item<float>();
                }
            }
        }
    }

    return output;
}

torch::Tensor forward(
    torch::Tensor x,
    double scaling_factor,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps,
    double bn_momentum
) {
    // Perform convolution
    x = conv2d_cpu(x, conv_weight, conv_bias);

    // Perform batch normalization
    x = batch_norm_cpu(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_momentum, bn_eps);

    // Scale the output
    x = x * scaling_factor;

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Performs convolution, batch normalization, and scaling on input tensor");
}
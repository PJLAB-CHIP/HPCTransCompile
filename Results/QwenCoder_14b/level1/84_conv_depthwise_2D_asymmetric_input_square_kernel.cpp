#include <torch/extension.h>
#include <omp.h>

// Define tile dimensions for the output tile
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

void depthwise_conv2d_unroll_gridstride_cpu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            // Map output channel back to input channel and its corresponding weight subgroup
            int in_ch = oc / channels_per_group;
            int weight_ch = oc % channels_per_group;

            // Determine the starting coordinates of the output tile processed by this block
            for (int tile_out_y = 0; tile_out_y < output_h; tile_out_y += TILE_HEIGHT) {
                for (int tile_out_x = 0; tile_out_x < output_w; tile_out_x += TILE_WIDTH) {
                    // Compute the corresponding top-left input coordinate for this output tile
                    int in_start_x = tile_out_x * stride - padding;
                    int in_start_y = tile_out_y * stride - padding;

                    // Load the weight kernel into a local array
                    float s_weight[kernel_size * kernel_size];
                    for (int i = 0; i < kernel_size * kernel_size; ++i) {
                        s_weight[i] = weight[
                            in_ch * (channels_per_group * kernel_size * kernel_size) +
                            weight_ch * (kernel_size * kernel_size) + i
                        ];
                    }

                    // Process each element in the tile
                    for (int local_y = 0; local_y < TILE_HEIGHT; ++local_y) {
                        for (int local_x = 0; local_x < TILE_WIDTH; ++local_x) {
                            int out_x = tile_out_x + local_x;
                            int out_y = tile_out_y + local_y;

                            if (out_x < output_w && out_y < output_h) {
                                float sum = 0.0f;
                                // Compute convolution sum by accessing input with given stride
                                if (kernel_size == 3) {
                                    // Manual unrolling for 3x3 kernel
                                    int s_y = local_y * stride;
                                    int s_x = local_x * stride;
                                    if (s_y >= 0 && s_y < input_h && s_x >= 0 && s_x < input_w) {
                                        sum += input[b * (in_channels * input_h * input_w) +
                                                        in_ch * (input_h * input_w) +
                                                        s_y * input_w + s_x] * s_weight[0];
                                    }
                                    if (s_y >= 0 && s_y < input_h && s_x + 1 >= 0 && s_x + 1 < input_w) {
                                        sum += input[b * (in_channels * input_h * input_w) +
                                                        in_ch * (input_h * input_w) +
                                                        s_y * input_w + s_x + 1] * s_weight[1];
                                    }
                                    if (s_y >= 0 && s_y < input_h && s_x + 2 >= 0 && s_x + 2 < input_w) {
                                        sum += input[b * (in_channels * input_h * input_w) +
                                                        in_ch * (input_h * input_w) +
                                                        s_y * input_w + s_x + 2] * s_weight[2];
                                    }

                                    if (s_y + 1 >= 0 && s_y + 1 < input_h && s_x >= 0 && s_x < input_w) {
                                        sum += input[b * (in_channels * input_h * input_w) +
                                                        in_ch * (input_h * input_w) +
                                                        (s_y + 1) * input_w + s_x] * s_weight[3];
                                    }
                                    if (s_y + 1 >= 0 && s_y + 1 < input_h && s_x + 1 >= 0 && s_x + 1 < input_w) {
                                        sum += input[b * (in_channels * input_h * input_w) +
                                                        in_ch * (input_h * input_w) +
                                                        (s_y + 1) * input_w + s_x + 1] * s_weight[4];
                                    }
                                    if (s_y + 1 >= 0 && s_y + 1 < input_h && s_x + 2 >= 0 && s_x + 2 < input_w) {
                                        sum += input[b * (in_channels * input_h * input_w) +
                                                        in_ch * (input_h * input_w) +
                                                        (s_y + 1) * input_w + s_x + 2] * s_weight[5];
                                    }

                                    if (s_y + 2 >= 0 && s_y + 2 < input_h && s_x >= 0 && s_x < input_w) {
                                        sum += input[b * (in_channels * input_h * input_w) +
                                                        in_ch * (input_h * input_w) +
                                                        (s_y + 2) * input_w + s_x] * s_weight[6];
                                    }
                                    if (s_y + 2 >= 0 && s_y + 2 < input_h && s_x + 1 >= 0 && s_x + 1 < input_w) {
                                        sum += input[b * (in_channels * input_h * input_w) +
                                                        in_ch * (input_h * input_w) +
                                                        (s_y + 2) * input_w + s_x + 1] * s_weight[7];
                                    }
                                    if (s_y + 2 >= 0 && s_y + 2 < input_h && s_x + 2 >= 0 && s_x + 2 < input_w) {
                                        sum += input[b * (in_channels * input_h * input_w) +
                                                        in_ch * (input_h * input_w) +
                                                        (s_y + 2) * input_w + s_x + 2] * s_weight[8];
                                    }
                                } else {
                                    // Use nested loops for other kernel sizes
                                    for (int ky = 0; ky < kernel_size; ++ky) {
                                        for (int kx = 0; kx < kernel_size; ++kx) {
                                            int s_y = local_y * stride + ky;
                                            int s_x = local_x * stride + kx;
                                            if (s_y >= 0 && s_y < input_h && s_x >= 0 && s_x < input_w) {
                                                sum += input[b * (in_channels * input_h * input_w) +
                                                                in_ch * (input_h * input_w) +
                                                                s_y * input_w + s_x] * s_weight[ky * kernel_size + kx];
                                            }
                                        }
                                    }
                                }
                                if (bias != nullptr) {
                                    sum += bias[oc];
                                }
                                int out_idx = b * (out_channels * output_h * output_w) +
                                              oc * (output_h * output_w) +
                                              out_y * output_w + out_x;
                                output[out_idx] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Forward function callable from Python
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(!input.is_cuda() && !weight.is_cuda(), "Input and weight must be CPU tensors");
    if (bias.has_value()) {
        TORCH_CHECK(!bias->is_cuda(), "Bias must be a CPU tensor if provided");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    depthwise_conv2d_unroll_gridstride_cpu(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with Manual Unrolling and Grid-Stride (CPU)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
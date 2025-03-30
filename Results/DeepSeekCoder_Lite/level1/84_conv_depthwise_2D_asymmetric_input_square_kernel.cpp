#include <torch/extension.h>
#include <omp.h>

// Define tile dimensions for the output tile
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

// This function performs the exact same operation as the CUDA kernel
torch::Tensor depthwise_conv2d_unroll_gridstride_shared_kernel(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::optional<torch::Tensor>& bias,
    int stride,
    int padding,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int channels_per_group
) {
    // Check the input tensors are on the correct device and have the correct types
    TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(weight.device().is_cpu(), "Weight must be a CPU tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().device().is_cpu(), "Bias must be a CPU tensor");
    }

    // Check the input tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_contiguous(), "Bias must be contiguous");
    }

    // Check the tensor dimensions
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    // Define the output tensor
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Define the grid and block dimensions for the kernel launch
    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);

    // Define the block size for the kernel launch
    int blockSize = 256;
    dim3 block(blockSize);

    // Calculate the required shared memory size
    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_width * smem_height + kernel_size * kernel_size) * sizeof(float);

    // Define the bias pointer
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    // Launch the kernel
    #pragma omp parallel for
    for (int bat_oc = 0; bat_oc < batch_size * out_channels; ++bat_oc) {
        int b = bat_oc / out_channels;
        int oc = bat_oc % out_channels;
        int in_ch = oc / channels_per_group;
        int weight_ch = oc % channels_per_group;
        int tile_out_x = (bat_oc / out_channels) * TILE_WIDTH;
        int tile_out_y = (bat_oc % out_channels) * TILE_HEIGHT;
        int in_start_x = tile_out_x * stride - padding;
        int in_start_y = tile_out_y * stride - padding;
        int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
        int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;
        float* s_input  = new float[smem_height * smem_width + kernel_size * kernel_size];
        float* s_weight = s_input + smem_height * smem_width;

        // Load the weight kernel into shared memory
        for (int i = 0; i < kernel_size * kernel_size; ++i) {
            s_weight[i] = weight[in_ch * (channels_per_group * kernel_size * kernel_size) +
                                  weight_ch * (kernel_size * kernel_size) + i];
        }

        // Load the required input patch into shared memory using a grid-stride loop
        for (int i = 0; i < smem_height * smem_width; ++i) {
            int r = i / smem_width;
            int c = i % smem_width;
            int global_y = in_start_y + r;
            int global_x = in_start_x + c;
            float val = 0.0f;
            if (global_y >= 0 && global_y < input_h && global_x >= 0 && global_x < input_w) {
                int input_idx = b * (in_channels * input_h * input_w) +
                                in_ch * (input_h * input_w) +
                                global_y * input_w + global_x;
                val = input[input_idx];
            }
            s_input[i] = val;
        }

        // Synchronize to ensure all threads have loaded the shared memory
        #pragma omp critical
        {
            // Compute the output tile
            int tile_area = TILE_WIDTH * TILE_HEIGHT;
            for (int i = 0; i < tile_area; ++i) {
                int local_y = i / TILE_WIDTH;
                int local_x = i % TILE_WIDTH;
                int out_x = tile_out_x + local_x;
                int out_y = tile_out_y + local_y;
                if (out_x < output_w && out_y < output_h) {
                    float sum = 0.0f;
                    // Compute convolution sum by accessing shared memory (input patch) with given stride
                    if (kernel_size == 3) {
                        // Manual unrolling for 3x3 kernel
                        sum += s_input[(local_y * stride) * smem_width + (local_x * stride)] * s_weight[0];
                        sum += s_input[(local_y * stride) * smem_width + (local_x * stride + 1)] * s_weight[1];
                        sum += s_input[(local_y * stride) * smem_width + (local_x * stride + 2)] * s_weight[2];

                        sum += s_input[((local_y * stride) + 1) * smem_width + (local_x * stride)] * s_weight[3];
                        sum += s_input[((local_y * stride) + 1) * smem_width + (local_x * stride + 1)] * s_weight[4];
                        sum += s_input[((local_y * stride) + 1) * smem_width + (local_x * stride + 2)] * s_weight[5];

                        sum += s_input[((local_y * stride) + 2) * smem_width + (local_x * stride)] * s_weight[6];
                        sum += s_input[((local_y * stride) + 2) * smem_width + (local_x * stride + 1)] * s_weight[7];
                        sum += s_input[((local_y * stride) + 2) * smem_width + (local_x * stride + 2)] * s_weight[8];
                    } else {
                        // Use #pragma unroll for other kernel sizes
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int s_y = local_y * stride + ky;
                                int s_x = local_x * stride + kx;
                                sum += s_input[s_y * smem_width + s_x] * s_weight[ky * kernel_size + kx];
                            }
                        }
                    }
                    if (bias_ptr != nullptr) {
                        sum += bias_ptr[oc];
                    }
                    int out_idx = b * (out_channels * output_h * output_w) +
                                  oc * (output_h * output_w) +
                                  out_y * output_w + out_x;
                    output[out_idx] = sum;
                }
            }
            delete[] s_input;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthwise_conv2d_unroll_gridstride_shared_kernel,
          "Depthwise 2D Convolution with Manual Unrolling and Grid-Stride (CPU)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
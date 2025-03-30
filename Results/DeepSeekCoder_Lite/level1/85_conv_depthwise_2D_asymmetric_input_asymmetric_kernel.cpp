#include <torch/extension.h>
#include <omp.h>
#include <vector>

// Tunable parameters
#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4

// Combined CPU kernel using shared memory tiling and vectorized processing

// Note: We assume the input tensor is laid out as (batch, channels, height, width).

void combined_depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group,
    int num_threads
) {
    // Calculate batch and channel from blockIdx.z (each block in z corresponds to one output channel of one batch element)
    int b, c, g, m;
    int tile_out_width = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    int tile_out_height = BLOCK_SIZE;
    int shared_tile_width = tile_out_width * stride_w + (kernel_w - 1) * dilation_w;
    int shared_tile_height = tile_out_height * stride_h + (kernel_h - 1) * dilation_h;

    // Parallelize over output channels
    #pragma omp parallel for private(b, c, g, m, tile_out_width, tile_out_height, shared_tile_width, shared_tile_height, base_in_x, base_in_y, shared_size, tidx, tidy, thread_id, sh_y, sh_x, in_y, in_x, val, results, sh_y, sh_x, in_val, res, out_y, base_out_x, results, j, out_idx) schedule(dynamic) num_threads(num_threads)
    for (b = 0; b < batch_size; ++b) {
        for (c = 0; c < out_channels; ++c) {
            g = c / channels_per_group; // group index
            m = c % channels_per_group; // channel index within group
            int base_in_x = -padding_w;
            int base_in_y = -padding_h;
            int shared_size = shared_tile_width * shared_tile_height;

            // Allocate shared memory (dynamically allocated by the kernel launch)
            std::vector<float> shared_input(shared_size);

            // Load the required input tile into shared memory
            #pragma omp for
            for (tidx = 0; tidx < BLOCK_SIZE; ++tidx) {
                for (tidy = 0; tidy < BLOCK_SIZE; ++tidy) {
                    thread_id = tidy * BLOCK_SIZE + tidx;
                    for (int idx = thread_id; idx < shared_size; idx += BLOCK_SIZE * BLOCK_SIZE) {
                        int sh_y = idx / shared_tile_width;
                        int sh_x = idx % shared_tile_width;
                        int in_y = base_in_y + sh_y;
                        int in_x = base_in_x + sh_x;
                        float val = 0.f;
                        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            val = input[((b * in_channels + g) * in_h + in_y) * in_w + in_x];
                        }
                        shared_input[sh_y * shared_tile_width + sh_x] = val;
                    }
                }
            }
            __sync_sub_and_fetch(&shared_input[0], 0); // Synchronize threads

            // Each thread computes ELEMENTS_PER_THREAD output pixels along the horizontal dimension for one row
            int out_y = 0;
            for (tidy = 0; tidy < BLOCK_SIZE; ++tidy) {
                out_y = blockIdx.y * BLOCK_SIZE + tidy;
                if (out_y < out_h) {
                    int base_out_x = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD + tidx * ELEMENTS_PER_THREAD;
                    std::vector<float> results(ELEMENTS_PER_THREAD, 0.f);

                    // Loop over kernel height and width
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int sh_y = tidy * stride_h + kh * dilation_h;
                            int base_sh_x = tidx * ELEMENTS_PER_THREAD * stride_w + kw * dilation_w;
                            for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
                                int sh_x = base_sh_x + j * stride_w;
                                float in_val = shared_input[sh_y * shared_tile_width + sh_x];
                                results[j] += in_val * weight[((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw];
                            }
                        }
                    }

                    // Write the computed outputs to global memory
                    for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
                        int out_x = base_out_x + j;
                        if (out_x < out_w) {
                            float res = results[j];
                            if (bias != nullptr) {
                                res += bias[c];
                            }
                            int out_idx = ((b * out_channels + c) * out_h + out_y) * out_w + out_x;
                            output[out_idx] = res;
                        }
                    }
                }
            }
        }
    }
}

// Host forward function

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    // Compute output dimensions
    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    // Grid configuration
    // Our block dimensions: (BLOCK_SIZE, BLOCK_SIZE)
    // Each block computes a tile of output of size:
    // Width: BLOCK_SIZE * ELEMENTS_PER_THREAD, Height: BLOCK_SIZE
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (out_w + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD),
        (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * out_channels
    );

    // Calculate required shared memory size
    int tile_out_width = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    int tile_out_height = BLOCK_SIZE;
    int shared_tile_width = tile_out_width * stride_w + (kernel_w - 1) * dilation_w;
    int shared_tile_height = tile_out_height * stride_h + (kernel_h - 1) * dilation_h;
    int shared_mem_size = shared_tile_width * shared_tile_height * sizeof(float);

    combined_depthwise_conv2d_kernel(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group,
        omp_get_max_threads()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined Depthwise Conv2D forward (CPU) with shared memory tiling and vectorized output");
}
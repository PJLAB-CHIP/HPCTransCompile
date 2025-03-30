#include <torch/extension.h>
#include <vector>
#include <math.h>
#include <omp.h>

// Dynamic block size selection based on channel count
inline int get_optimal_block_size(int channels) {
    if (channels <= 32) return 128;
    else if (channels <= 64) return 256;
    else return 512;
}

template <typename scalar_t, int CHANNELS>
void adaptive_block_softmax_sigmoid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch,
    const int depth,
    const int height,
    const int width,
    const int block_size,
    const int shared_mem_size) {

    const int spatial = depth * height * width;
    const int total_pixels = batch * spatial;

    #pragma omp parallel for
    for (int idx = 0; idx < total_pixels; ++idx) {
        const int b = idx / spatial;
        const int pixel_idx = idx % spatial;
        const int d = pixel_idx / (height * width);
        const int rem = pixel_idx % (height * width);
        const int h = rem / width;
        const int w = rem % width;

        const int base = (b * CHANNELS * spatial) + (d * height * width + h * width + w);
        const int stride = spatial;

        // Load initial values into shared memory
        scalar_t local_max = -INFINITY;
        scalar_t local_vals[8];  // Cache frequently accessed values

        #pragma unroll
        for (int c = 0; c < CHANNELS; c += 8) {
            #pragma unroll
            for (int u = 0; u < 8 && (c + u) < CHANNELS; ++u) {
                local_vals[u] = input[base + (c + u) * stride];
                local_max = std::max(local_max, local_vals[u]);
            }
        }

        // Compute sum of exponentials
        scalar_t sum_exp = 0.0f;

        #pragma unroll
        for (int c = 0; c < CHANNELS; c += 8) {
            #pragma unroll
            for (int u = 0; u < 8 && (c + u) < CHANNELS; ++u) {
                sum_exp += exp(local_vals[u] - local_max);
            }
        }

        // Compute final softmax and sigmoid values
        #pragma unroll
        for (int c = 0; c < CHANNELS; c += 8) {
            #pragma unroll
            for (int u = 0; u < 8 && (c + u) < CHANNELS; ++u) {
                const int pos = base + (c + u) * stride;
                const scalar_t softmax_val = exp(input[pos] - local_max) / sum_exp;
                output[pos] = 1.0f / (1.0f + exp(-softmax_val));
            }
        }
    }
}

template <typename scalar_t>
void dynamic_adaptive_block_softmax_sigmoid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int channels,
    const int batch,
    const int depth,
    const int height,
    const int width,
    const int block_size,
    const int shared_mem_size) {

    const int spatial = depth * height * width;
    const int total_pixels = batch * spatial;

    #pragma omp parallel for
    for (int idx = 0; idx < total_pixels; ++idx) {
        const int b = idx / spatial;
        const int pixel_idx = idx % spatial;
        const int d = pixel_idx / (height * width);
        const int rem = pixel_idx % (height * width);
        const int h = rem / width;
        const int w = rem % width;

        const int base = (b * channels * spatial) + (d * height * width + h * width + w);
        const int stride = spatial;

        scalar_t local_max = -INFINITY;
        for (int c = 0; c < channels; ++c) {
            local_max = std::max(local_max, input[base + c * stride]);
        }

        // Compute sum of exponentials
        scalar_t sum_exp = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sum_exp += exp(input[base + c * stride] - local_max);
        }

        // Compute final softmax and sigmoid values
        for (int c = 0; c < channels; ++c) {
            const int pos = base + c * stride;
            const scalar_t softmax_val = exp(input[pos] - local_max) / sum_exp;
            output[pos] = 1.0f / (1.0f + exp(-softmax_val));
        }
    }
}

torch::Tensor forward(
    torch::Tensor input,
    int stride,
    int padding,
    int output_padding,
    bool bias_flag,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias) {

    auto x = torch::conv_transpose3d(
        input,
        conv_transpose,
        bias_flag ? conv_transpose_bias : torch::Tensor(),
        stride,
        padding,
        output_padding
    );

    const int batch = x.size(0);
    const int channels = x.size(1);
    const int depth = x.size(2);
    const int height = x.size(3);
    const int width = x.size(4);

    auto output = torch::empty_like(x);
    
    const int block_size = get_optimal_block_size(channels);
    const int shared_mem_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "adaptive_block_softmax_sigmoid_kernel", ([&] {
        if (channels == 32) {
            adaptive_block_softmax_sigmoid_kernel<scalar_t, 32>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch,
                depth,
                height,
                width,
                block_size,
                shared_mem_size);
        } else if (channels == 64) {
            adaptive_block_softmax_sigmoid_kernel<scalar_t, 64>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch,
                depth,
                height,
                width,
                block_size,
                shared_mem_size);
        } else {
            dynamic_adaptive_block_softmax_sigmoid_kernel<scalar_t>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                channels,
                batch,
                depth,
                height,
                width,
                block_size,
                shared_mem_size);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Block Size Softmax Sigmoid Forward");
}
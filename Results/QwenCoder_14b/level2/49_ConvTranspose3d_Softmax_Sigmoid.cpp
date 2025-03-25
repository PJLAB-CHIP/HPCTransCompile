#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

inline int get_optimal_block_size(int channels) {
    if (channels <= 32) return 128;
    else if (channels <= 64) return 256;
    else return 512;
}

template <typename scalar_t, int CHANNELS>
void adaptive_block_softmax_sigmoid_cpu(
    const scalar_t* input,
    scalar_t* output,
    const int batch,
    const int depth,
    const int height,
    const int width) {

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

        scalar_t local_max = -INFINITY;
        scalar_t local_vals[8];

        #pragma unroll
        for (int c = 0; c < CHANNELS; c += 8) {
            #pragma unroll
            for (int u = 0; u < 8 && (c + u) < CHANNELS; ++u) {
                local_vals[u] = input[base + (c + u) * stride];
                local_max = std::max(local_max, local_vals[u]);
            }
        }

        scalar_t sum_exp = 0.0f;

        #pragma unroll
        for (int c = 0; c < CHANNELS; c += 8) {
            #pragma unroll
            for (int u = 0; u < 8 && (c + u) < CHANNELS; ++u) {
                sum_exp += exp(local_vals[u] - local_max);
            }
        }

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
void dynamic_adaptive_block_softmax_sigmoid_cpu(
    const scalar_t* input,
    scalar_t* output,
    const int channels,
    const int batch,
    const int depth,
    const int height,
    const int width) {

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

        #pragma unroll 4
        for (int c = 0; c < channels; ++c) {
            local_max = std::max(local_max, input[base + c * stride]);
        }

        scalar_t sum_exp = 0.0f;

        #pragma unroll 4
        for (int c = 0; c < channels; ++c) {
            sum_exp += exp(input[base + c * stride] - local_max);
        }

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
    const int total_pixels = batch * depth * height * width;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "adaptive_block_softmax_sigmoid_cpu", ([&] {
        if (channels == 32) {
            adaptive_block_softmax_sigmoid_cpu<scalar_t, 32>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch,
                depth,
                height,
                width);
        } else if (channels == 64) {
            adaptive_block_softmax_sigmoid_cpu<scalar_t, 64>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch,
                depth,
                height,
                width);
        } else {
            dynamic_adaptive_block_softmax_sigmoid_cpu<scalar_t>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                channels,
                batch,
                depth,
                height,
                width);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Block Size Softmax Sigmoid Forward");
}
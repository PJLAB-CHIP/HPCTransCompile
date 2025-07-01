#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel assumes a fixed 3x3 convolution and 2x2 max pooling.
// Manually unroll inner loops to reduce loop overhead and improve performance.

__global__ void manually_unrolled_kernel(
    const float* __restrict__ input,    // [batch_size, in_channels, height, width]
    const float* __restrict__ weight,   // [out_channels, in_channels, 3, 3]
    const float* __restrict__ bias,     // [out_channels]
    float* __restrict__ output,         // [batch_size, out_channels, pooled_h, pooled_w]
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const float subtract_val,
    const int out_h,   // out_h = height - 3 + 1
    const int out_w,   // out_w = width  - 3 + 1
    const int pooled_h,
    const int pooled_w
) {
    // Compute 2D spatial indices for pooling and combine batch/channel in z-dimension
    int x = blockIdx.x * blockDim.x + threadIdx.x; // pooled width index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // pooled height index
    int bc = blockIdx.z; // combined batch and channel index
    int batch = bc / out_channels;
    int channel = bc % out_channels;

    if (x >= pooled_w || y >= pooled_h || batch >= batch_size) return;

    // For this manually unrolled kernel, we assume pooling kernel size is 2
    int h_start = y * 2;
    int w_start = x * 2;
    float max_val = -1e10f;

    // Manually unrolled 2x2 pooling with four fixed offsets
    // Pool offset (0,0)
    int cur_h = h_start;
    int cur_w = w_start;
    if (cur_h < out_h && cur_w < out_w) {
        float conv = bias[channel];
        // Manually unroll the 3x3 convolution for each in_channel
        for (int ic = 0; ic < in_channels; ic++) {
            int weight_base = ((channel * in_channels) + ic) * 9; // 3x3 = 9
            int input_base = ((batch * in_channels) + ic) * height * width;

            conv += input[input_base + (cur_h + 0) * width + (cur_w + 0)] * weight[weight_base + 0];
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 1)] * weight[weight_base + 1];
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 2)] * weight[weight_base + 2];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 0)] * weight[weight_base + 3];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 1)] * weight[weight_base + 4];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 2)] * weight[weight_base + 5];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 0)] * weight[weight_base + 6];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 1)] * weight[weight_base + 7];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 2)] * weight[weight_base + 8];
        }
        conv -= subtract_val;
        float tmp = conv + 3.0f;
        float clamp_val = fminf(6.0f, fmaxf(0.0f, tmp));
        float hardswish = conv * clamp_val / 6.0f;
        max_val = fmaxf(max_val, hardswish);
    }

    // Pool offset (0,1)
    cur_h = h_start;
    cur_w = w_start + 1;
    if (cur_h < out_h && cur_w < out_w) {
        float conv = bias[channel];
        for (int ic = 0; ic < in_channels; ic++) {
            int weight_base = ((channel * in_channels) + ic) * 9;
            int input_base = ((batch * in_channels) + ic) * height * width;
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 0)] * weight[weight_base + 0];
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 1)] * weight[weight_base + 1];
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 2)] * weight[weight_base + 2];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 0)] * weight[weight_base + 3];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 1)] * weight[weight_base + 4];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 2)] * weight[weight_base + 5];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 0)] * weight[weight_base + 6];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 1)] * weight[weight_base + 7];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 2)] * weight[weight_base + 8];
        }
        conv -= subtract_val;
        float tmp = conv + 3.0f;
        float clamp_val = fminf(6.0f, fmaxf(0.0f, tmp));
        float hardswish = conv * clamp_val / 6.0f;
        max_val = fmaxf(max_val, hardswish);
    }

    // Pool offset (1,0)
    cur_h = h_start + 1;
    cur_w = w_start;
    if (cur_h < out_h && cur_w < out_w) {
        float conv = bias[channel];
        for (int ic = 0; ic < in_channels; ic++) {
            int weight_base = ((channel * in_channels) + ic) * 9;
            int input_base = ((batch * in_channels) + ic) * height * width;
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 0)] * weight[weight_base + 0];
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 1)] * weight[weight_base + 1];
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 2)] * weight[weight_base + 2];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 0)] * weight[weight_base + 3];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 1)] * weight[weight_base + 4];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 2)] * weight[weight_base + 5];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 0)] * weight[weight_base + 6];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 1)] * weight[weight_base + 7];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 2)] * weight[weight_base + 8];
        }
        conv -= subtract_val;
        float tmp = conv + 3.0f;
        float clamp_val = fminf(6.0f, fmaxf(0.0f, tmp));
        float hardswish = conv * clamp_val / 6.0f;
        max_val = fmaxf(max_val, hardswish);
    }

    // Pool offset (1,1)
    cur_h = h_start + 1;
    cur_w = w_start + 1;
    if (cur_h < out_h && cur_w < out_w) {
        float conv = bias[channel];
        for (int ic = 0; ic < in_channels; ic++) {
            int weight_base = ((channel * in_channels) + ic) * 9;
            int input_base = ((batch * in_channels) + ic) * height * width;
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 0)] * weight[weight_base + 0];
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 1)] * weight[weight_base + 1];
            conv += input[input_base + (cur_h + 0) * width + (cur_w + 2)] * weight[weight_base + 2];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 0)] * weight[weight_base + 3];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 1)] * weight[weight_base + 4];
            conv += input[input_base + (cur_h + 1) * width + (cur_w + 2)] * weight[weight_base + 5];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 0)] * weight[weight_base + 6];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 1)] * weight[weight_base + 7];
            conv += input[input_base + (cur_h + 2) * width + (cur_w + 2)] * weight[weight_base + 8];
        }
        conv -= subtract_val;
        float tmp = conv + 3.0f;
        float clamp_val = fminf(6.0f, fmaxf(0.0f, tmp));
        float hardswish = conv * clamp_val / 6.0f;
        max_val = fmaxf(max_val, hardswish);
    }

    // Apply Mish activation: x * tanh(softplus(x))
    float softplus = logf(1.0f + expf(max_val));
    float mish = max_val * tanhf(softplus);

    int out_idx = ((batch * out_channels + channel) * pooled_h + y) * pooled_w + x;
    output[out_idx] = mish;
}


// Host function launching the manually unrolled kernel
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    int pool_kernel_size  // Expected to be 2 for this unrolled kernel
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2); // should be 3

    const int out_h = height - kernel_size + 1;
    const int out_w = width - kernel_size + 1;
    const int pooled_h = (out_h + pool_kernel_size - 1) / pool_kernel_size;
    const int pooled_w = (out_w + pool_kernel_size - 1) / pool_kernel_size;

    auto output = torch::empty({batch_size, out_channels, pooled_h, pooled_w},
                               torch::TensorOptions().device(input.device()).dtype(input.dtype()));

    // Configure 2D block dimensions and 3D grid dimensions
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (pooled_w + block_dim.x - 1) / block_dim.x,
        (pooled_h + block_dim.y - 1) / block_dim.y,
        batch_size * out_channels
    );

    manually_unrolled_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        subtract_value,
        out_h,
        out_w,
        pooled_h,
        pooled_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fully manually unrolled conv-pool-activate forward (CUDA)");
}

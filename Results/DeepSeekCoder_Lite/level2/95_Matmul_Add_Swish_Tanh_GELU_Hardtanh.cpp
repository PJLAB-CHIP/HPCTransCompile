#include <torch/extension.h>
#include <math.h>
#include <omp.h>

#define WARP_SIZE 32

// Host function for launching the kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_value) {

    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda() && add_value.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && 
                bias.is_contiguous() && add_value.is_contiguous(),
                "All inputs must be contiguous");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, x.options());

    int num_outputs = batch_size * out_features; // each output element computed by one warp
    int warps_per_block = 4; // e.g. 128 threads per block => 4 warps per block
    int threads_per_block = warps_per_block * WARP_SIZE;
    int num_blocks = (num_outputs + threads_per_block - 1) / threads_per_block;

    #pragma omp parallel for
    for (int warp_global_id = 0; warp_global_id < num_outputs; ++warp_global_id) {
        int i = warp_global_id / out_features;
        int j = warp_global_id % out_features;

        int lane = omp_get_thread_num() % WARP_SIZE;
        float sum = 0.0f;

        // Compute base addresses
        int base_x = i * in_features;
        int base_w = j * in_features;

        // Use vectorized loads when possible: process 4 floats (128 bits) at a time
        int num_vec_iters = in_features / 4;
        int rem = in_features % 4;

        // Reinterpret pointers for aligned float4 loads
        const float* x_vec = x.data_ptr<float>() + base_x;
        const float* w_vec = weight.data_ptr<float>() + base_w;

        // Each thread in the warp processes several float4 elements
        for (int idx = lane; idx < num_vec_iters; idx += WARP_SIZE) {
            float x_val = x_vec[idx * 4 + 0];
            float w_val = w_vec[idx * 4 + 0];
            sum += x_val * w_val;
            x_val = x_vec[idx * 4 + 1];
            w_val = w_vec[idx * 4 + 1];
            sum += x_val * w_val;
            x_val = x_vec[idx * 4 + 2];
            w_val = w_vec[idx * 4 + 2];
            sum += x_val * w_val;
            x_val = x_vec[idx * 4 + 3];
            w_val = w_vec[idx * 4 + 3];
            sum += x_val * w_val;
        }

        // Process remaining elements if in_features is not a multiple of 4
        int rem_start = num_vec_iters * 4;
        for (int k = rem_start + lane; k < in_features; k += WARP_SIZE) {
            float xv = x_vec[k];
            float wv = w_vec[k];
            sum += xv * wv;
        }

        // Warp-level reduction using __shfl_down_sync
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Lane 0 applies bias, add_value and activation functions
        if (lane == 0) {
            sum += bias[j];
            sum += add_value[j];

            // Swish activation
            float sigmoid = 1.0f / (1.0f + expf(-sum));
            sum *= sigmoid;

            // Tanh activation
            sum = tanhf(sum);

            // GELU activation
            sum = 0.5f * sum * (1.0f + erff(sum / 1.41421356237f));

            // Hardtanh activation
            sum = fmaxf(fminf(sum, 1.0f), -1.0f);

            output[i * out_features + j] = sum;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level forward CUDA kernel with vectorized __ldg() loads");
}
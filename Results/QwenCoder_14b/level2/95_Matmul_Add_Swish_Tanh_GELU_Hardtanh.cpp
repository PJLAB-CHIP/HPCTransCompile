#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Function to perform the forward computation on CPU
torch::Tensor forward_cpu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_value) {

    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && 
                bias.is_contiguous() && add_value.is_contiguous(),
                "All inputs must be contiguous");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, x.options());

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_features; ++j) {
            float sum = 0.0f;

            // Compute base addresses
            int base_x = i * in_features;
            int base_w = j * in_features;

            // Use vectorized loads when possible: process 4 floats (128 bits) at a time
            int num_vec_iters = in_features / 4;
            int rem = in_features % 4;

            // Reinterpret pointers for aligned float4 loads
            const float4* x_vec = reinterpret_cast<const float4*>(x.data_ptr<float>() + base_x);
            const float4* w_vec = reinterpret_cast<const float4*>(weight.data_ptr<float>() + base_w);

            // Each thread in the warp processes several float4 elements
            for (int idx = 0; idx < num_vec_iters; ++idx) {
                float4 x_val = x_vec[idx];
                float4 w_val = w_vec[idx];
                sum += x_val.x * w_val.x + x_val.y * w_val.y + x_val.z * w_val.z + x_val.w * w_val.w;
            }

            // Process remaining elements if in_features is not a multiple of 4
            int rem_start = num_vec_iters * 4;
            for (int k = rem_start; k < in_features; ++k) {
                float xv = x.data_ptr<float>()[base_x + k];
                float wv = weight.data_ptr<float>()[base_w + k];
                sum += xv * wv;
            }

            // Apply bias, add_value and activation functions
            sum += bias.data_ptr<float>()[j];
            sum += add_value.data_ptr<float>()[j];

            // Swish activation
            float sigmoid = 1.0f / (1.0f + expf(-sum));
            sum *= sigmoid;

            // Tanh activation
            sum = tanhf(sum);

            // GELU activation
            sum = 0.5f * sum * (1.0f + erf(sum / 1.41421356237f));

            // Hardtanh activation
            sum = fmaxf(fminf(sum, 1.0f), -1.0f);

            output.data_ptr<float>()[i * out_features + j] = sum;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "CPU forward function with vectorized loads");
}
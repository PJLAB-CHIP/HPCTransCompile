#include <torch/extension.h>
#include <omp.h>

#include <vector>

__device__ float compute_dot_product(const float* x_row, const float* weight_row, int in_features) {
    float sum = 0.0f;
    for (int j = 0; j < in_features; ++j) {
        sum += x_row[j] * weight_row[j];
    }
    return sum;
}

__device__ float apply_min_subtract(float computed, float bias, float constant) {
    float result = computed + bias;  // Add bias
    if (result > constant) {  // Min with constant
        result = constant;
    }
    result -= constant;  // Subtract constant
    return result;
}

__global__ void my_kernel(
    const float* x,
    const float* linear_weight,
    const float* linear_bias,
    const float* constant,
    float* y,
    int batch_size,
    int in_features,
    int out_features) {
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x;
    
    if (out_idx < out_features) {
        // Pointers to rows
        const float* x_row = x + batch_idx * in_features;
        const float* weight_row = linear_weight + out_idx * in_features;
        float bias = linear_bias[out_idx];
        float cst = constant[out_idx];

        // Compute dot product using device function
        float result = compute_dot_product(x_row, weight_row, in_features);

        // Apply min and subtract using another device function
        y[batch_idx * out_features + out_idx] = apply_min_subtract(result, bias, cst);
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor linear_weight,
    torch::Tensor linear_bias,
    torch::Tensor constant) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(linear_weight.is_cuda(), "linear_weight must be a CUDA tensor");
    TORCH_CHECK(linear_bias.is_cuda(), "linear_bias must be a CUDA tensor");
    TORCH_CHECK(constant.is_cuda(), "constant must be a CUDA tensor");

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = linear_weight.size(0);

    auto y = torch::zeros({batch_size, out_features}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = linear_weight.data_ptr<float>();
    const float* bias_ptr = linear_bias.data_ptr<float>();
    const float* constant_ptr = constant.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    dim3 grid_size(batch_size);
    dim3 block_size(out_features);

    my_kernel<<<grid_size, block_size>>>(
        x_ptr,
        weight_ptr,
        bias_ptr,
        constant_ptr,
        y_ptr,
        batch_size,
        in_features,
        out_features);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function");
}
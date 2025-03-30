#include <torch/extension.h>
#include <vector>
#include <omp.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__forceinline__ float4 load_float4(const float* addr) {
    float4 val;
    val.x = addr[0];
    val.y = addr[1];
    val.z = addr[2];
    val.w = addr[3];
    return val;
}

__forceinline__ void store_float4(float* addr, float4 val) {
    addr[0] = val.x;
    addr[1] = val.y;
    addr[2] = val.z;
    addr[3] = val.w;
}

__forceinline__ float process_value(float val, const float* bias, int bias_idx) {
    // ReLU
    val = std::fmaxf(0.0f, val);
    
    // LeakyReLU
    val = std::fmaxf(0.01f * val, val);
    
    // GELU
    const float sqrt_2_over_pi = std::sqrtf(2.0f / M_PI);
    val = 0.5f * val * (1.0f + std::tanh(sqrt_2_over_pi * (val + 0.044715f * std::powf(val, 3.0f))));
    
    // Sigmoid
    val = 1.0f / (1.0f + std::expf(-val));
    
    // Add bias
    val += bias[bias_idx];
    
    return val;
}

torch::Tensor apply_activations_and_bias_cpu(
    torch::Tensor output, const torch::Tensor& bias,
    int batch_size, int out_channels, int depth, int height, int width
) {
    int total_elements = batch_size * out_channels * depth * height * width;
    float* output_data = output.data_ptr<float>();

    #pragma omp parallel for
    for (int i = 0; i < total_elements; ++i) {
        int bias_idx = (i / (depth * height * width)) % out_channels;
        output_data[i] = process_value(output_data[i], bias.data_ptr<float>(), bias_idx);
    }

    return output;
}

torch::Tensor module_fn_cpu(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(conv_weight);
    CHECK_INPUT(conv_bias);
    CHECK_INPUT(bias);

    auto output = torch::conv3d(x, conv_weight, conv_bias);

    return apply_activations_and_bias_cpu(output, bias, x.size(0), x.size(1), x.size(2), x.size(3), x.size(4));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu, "CPU implementation of module_fn");
}
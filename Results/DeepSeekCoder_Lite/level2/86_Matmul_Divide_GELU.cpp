#include <torch/extension.h>
#include <omp.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

// Device function for GELU calculation
__forceinline__ float gelu_impl(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x * x * x)));
    return x * cdf;
}

void block_optimized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight_t,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int input_size,
    const int output_size,
    const float divisor,
    int block_size,
    int num_blocks
) {
    #pragma omp parallel for
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        for (int row = block_idx * block_size; row < batch_size && row < (block_idx + 1) * block_size; ++row) {
            for (int col = 0; col < output_size; ++col) {
                float acc = bias[col];
                for (int tile = 0; tile < (input_size + block_size - 1) / block_size; ++tile) {
                    int input_row = row;
                    int input_col = tile * block_size + threadIdx.x;
                    int weight_row = tile * block_size + threadIdx.y;
                    int weight_col = col;

                    float input_value = (input_row < batch_size && input_col < input_size) ? x[input_row * input_size + input_col] : 0.0f;
                    float weight_value = (weight_row < input_size && weight_col < output_size) ? weight_t[weight_row * output_size + weight_col] : 0.0f;

                    acc += input_value * weight_value;
                }
                output[row * output_size + col] = gelu_impl(acc / divisor);
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    
    auto weight_t = weight.transpose(0, 1).contiguous();
    
    const int batch_size = x.size(0);
    const int input_size = x.size(1);
    const int output_size = weight.size(0);
    
    auto output = torch::empty({batch_size, output_size}, x.options());
    
    int block_size = BLOCK_SIZE;
    int num_blocks = (output_size + block_size - 1) / block_size * (batch_size + block_size - 1) / block_size;
    
    block_optimized_kernel(
        x.data_ptr<float>(),
        weight_t.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size,
        divisor,
        block_size,
        num_blocks
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Block size optimized fused kernel");
}
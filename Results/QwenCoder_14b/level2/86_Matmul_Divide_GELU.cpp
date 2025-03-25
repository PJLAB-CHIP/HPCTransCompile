#include <torch/extension.h>
#include <omp.h>
#include <cmath>

#define BLOCK_SIZE 16

// CPU function for GELU calculation
float gelu_impl(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x * x * x)));
    return x * cdf;
}

void block_optimized_cpu_kernel(
    const float* x,
    const float* weight_t,
    const float* bias,
    float* output,
    const int batch_size,
    const int input_size,
    const int output_size,
    const float divisor
) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < output_size; ++col) {
            float acc = bias[col];
            const int num_tiles = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            for (int tile = 0; tile < num_tiles; ++tile) {
                const int input_row = row;
                const int input_col = tile * BLOCK_SIZE;
                const int weight_row = tile * BLOCK_SIZE;
                const int weight_col = col;

                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    if (input_row < batch_size && input_col + k < input_size && weight_row + k < input_size && weight_col < output_size) {
                        acc += x[input_row * input_size + input_col + k] * weight_t[weight_row * output_size + weight_col];
                    }
                }
            }

            acc /= divisor;
            output[row * output_size + col] = gelu_impl(acc);
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
    
    block_optimized_cpu_kernel(
        x.data_ptr<float>(),
        weight_t.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size,
        divisor
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Block size optimized fused kernel");
}
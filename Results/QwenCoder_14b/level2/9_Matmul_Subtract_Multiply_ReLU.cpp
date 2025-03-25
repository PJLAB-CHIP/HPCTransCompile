#include <torch/extension.h>
#include <omp.h>

// CPU function for combined linear, subtract, multiply and ReLU operations
template <typename scalar_t>
void linear_subtract_multiply_relu_cpu(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float subtract_value,
    const float multiply_value) {

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < out_features; ++col) {
            scalar_t sum = 0;
            
            // Compute linear transformation
            for (int k = 0; k < in_features; k++) {
                sum += input[row * in_features + k] * weight[col * in_features + k];
            }
            
            // Add bias
            sum += bias[col];
            
            // Subtract and multiply
            sum = (sum - subtract_value) * multiply_value;
            
            // ReLU activation
            sum = sum > 0 ? sum : 0;
            
            output[row * out_features + col] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    linear_subtract_multiply_relu_cpu<scalar_t>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        in_features, 
        out_features,
        subtract_value,
        multiply_value
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Linear transform with subtract, multiply and ReLU forward");
}
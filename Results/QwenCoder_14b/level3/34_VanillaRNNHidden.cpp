#include <torch/extension.h>
#include <cmath>
#include <vector>

// Function to perform the RNN forward pass on CPU
void rnn_forward_cpu(
    const std::vector<float>& x,      // [batch, input_size]
    const std::vector<float>& h,      // [batch, hidden_size]
    const std::vector<float>& weight, // [hidden_dim, (input_size+hidden_size)]
    const std::vector<float>& bias,   // [hidden_dim]
    std::vector<float>& output,       // [batch, hidden_dim]
    int input_size,
    int hidden_size,
    int batch_size,
    int hidden_dim
) {
    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int neuron = 0; neuron < hidden_dim; ++neuron) {
            float local_sum = 0.0f;

            // Process input data
            for (int idx = 0; idx < input_size; ++idx) {
                local_sum += x[batch * input_size + idx] * weight[neuron * (input_size + hidden_size) + idx];
            }

            // Process hidden state data
            for (int idx = 0; idx < hidden_size; ++idx) {
                local_sum += h[batch * hidden_size + idx] * weight[neuron * (input_size + hidden_size) + input_size + idx];
            }

            // Add bias and apply tanh activation
            output[batch * hidden_dim + neuron] = tanhf(local_sum + bias[neuron]);
        }
    }
}

torch::Tensor module_fn(
    torch::Tensor x,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias,
    torch::Tensor hidden
) {
    x = x.contiguous();
    hidden = hidden.to(x.device()).contiguous();
    i2h_weight = i2h_weight.contiguous();
    i2h_bias = i2h_bias.contiguous();

    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = hidden.size(1);
    int hidden_dim = i2h_weight.size(0);

    auto output = torch::empty({batch_size, hidden_dim}, x.options());

    // Convert tensors to vectors for easier processing
    std::vector<float> x_vec(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
    std::vector<float> h_vec(hidden.data_ptr<float>(), hidden.data_ptr<float>() + hidden.numel());
    std::vector<float> weight_vec(i2h_weight.data_ptr<float>(), i2h_weight.data_ptr<float>() + i2h_weight.numel());
    std::vector<float> bias_vec(i2h_bias.data_ptr<float>(), i2h_bias.data_ptr<float>() + i2h_bias.numel());
    std::vector<float> output_vec(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());

    // Perform the RNN forward pass on CPU
    rnn_forward_cpu(
        x_vec,
        h_vec,
        weight_vec,
        bias_vec,
        output_vec,
        input_size,
        hidden_size,
        batch_size,
        hidden_dim
    );

    // Copy the result back to the tensor
    std::copy(output_vec.begin(), output_vec.end(), output.data_ptr<float>());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "RNN forward on CPU");
}
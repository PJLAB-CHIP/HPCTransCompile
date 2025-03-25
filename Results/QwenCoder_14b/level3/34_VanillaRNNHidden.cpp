#include <torch/extension.h>
#include <omp.h>
#include <cmath>

void rnn_forward_cpu(
    const float* x,      // [batch, input_size]
    const float* h,      // [batch, hidden_size]
    const float* weight, // [hidden_dim, (input_size+hidden_size)]
    const float* bias,    // [hidden_dim]
    float* output,        // [batch, hidden_dim]
    int batch_size,
    int input_size,
    int hidden_size
) {
    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int neuron = 0; neuron < hidden_size; ++neuron) {
            float local_sum = 0.0f;

            // Process input data
            for (int idx = 0; idx < input_size; ++idx) {
                local_sum += x[batch * input_size + idx] * weight[neuron * (input_size + hidden_size) + idx];
            }

            // Process hidden state data
            for (int idx = 0; idx < hidden_size; ++idx) {
                local_sum += h[batch * hidden_size + idx] * weight[neuron * (input_size + hidden_size) + input_size + idx];
            }

            // Apply bias and tanh activation
            output[batch * hidden_size + neuron] = tanhf(local_sum + bias[neuron]);
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

    auto output = torch::empty({batch_size, hidden_size}, x.options());

    rnn_forward_cpu(
        x.data_ptr<float>(),
        hidden.data_ptr<float>(),
        i2h_weight.data_ptr<float>(),
        i2h_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "RNN forward on CPU");
}
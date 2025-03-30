#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Optimized kernel using __ldg() for read-only data and aligned memory access
void rnn_forward_aligned_ldg_kernel(
    const float* x,      // [batch, input_size]
    const float* h,      // [batch, hidden_size]
    const float* weight, // [hidden_dim, (input_size+hidden_size)]
    const float* bias,   // [hidden_dim]
    float* output,       // [batch, hidden_dim]
    int input_size,
    int hidden_size,
    int batch,
    int hidden_size_batch
) {
    int combined_dim = (input_size + hidden_size);

    // Process input data with aligned float4 loads
    int input_blocks = (input_size + 3) / 4;
    int hidden_blocks = (hidden_size + 3) / 4;

    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
        for (int neuron_idx = 0; neuron_idx < hidden_size_batch; ++neuron_idx) {
            float local_sum = 0.0f;

            for (int idx = 0; idx < input_blocks; ++idx) {
                float4 val = {x[batch_idx * input_blocks + idx * 4],
                               x[batch_idx * input_blocks + idx * 4 + 1],
                               x[batch_idx * input_blocks + idx * 4 + 2],
                               x[batch_idx * input_blocks + idx * 4 + 3]};
                float4 w = {weight[neuron_idx * combined_dim + idx * 4],
                             weight[neuron_idx * combined_dim + idx * 4 + 1],
                             weight[neuron_idx * combined_dim + idx * 4 + 2],
                             weight[neuron_idx * combined_dim + idx * 4 + 3]};

                // Handle partial float4 at boundary
                if (idx == input_blocks - 1 && (input_size % 4) != 0) {
                    switch (input_size % 4) {
                        case 1:
                            local_sum += val.x * w.x;
                            break;
                        case 2:
                            local_sum += val.x * w.x + val.y * w.y;
                            break;
                        case 3:
                            local_sum += val.x * w.x + val.y * w.y + val.z * w.z;
                            break;
                    }
                } else {
                    local_sum += val.x * w.x + val.y * w.y + val.z * w.z + val.w * w.w;
                }
            }

            for (int idx = 0; idx < hidden_blocks; ++idx) {
                float4 val = {h[batch_idx * hidden_blocks + idx * 4],
                               h[batch_idx * hidden_blocks + idx * 4 + 1],
                               h[batch_idx * hidden_blocks + idx * 4 + 2],
                               h[batch_idx * hidden_blocks + idx * 4 + 3]};
                float4 w = {weight[neuron_idx * combined_dim + input_blocks * 4 + idx * 4],
                             weight[neuron_idx * combined_dim + input_blocks * 4 + idx * 4 + 1],
                             weight[neuron_idx * combined_dim + input_blocks * 4 + idx * 4 + 2],
                             weight[neuron_idx * combined_dim + input_blocks * 4 + idx * 4 + 3]};

                // Handle partial float4 at boundary
                if (idx == hidden_blocks - 1 && (hidden_size % 4) != 0) {
                    switch (hidden_size % 4) {
                        case 1:
                            local_sum += val.x * w.x;
                            break;
                        case 2:
                            local_sum += val.x * w.x + val.y * w.y;
                            break;
                        case 3:
                            local_sum += val.x * w.x + val.y * w.y + val.z * w.z;
                            break;
                    }
                } else {
                    local_sum += val.x * w.x + val.y * w.y + val.z * w.z + val.w * w.w;
                }
            }

            // Reduce within block using sequential addressing
            local_sum = tanhf(local_sum + bias[neuron_idx]);
            output[batch_idx * hidden_size_batch + neuron_idx] = local_sum;
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

    int batch = x.size(0);
    int input_size = x.size(1);
    int hidden_size = hidden.size(1);

    auto output = torch::empty({batch, hidden_size}, x.options());

    rnn_forward_aligned_ldg_kernel(
        x.data_ptr<float>(),
        hidden.data_ptr<float>(),
        i2h_weight.data_ptr<float>(),
        i2h_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        input_size,
        hidden_size,
        batch,
        hidden_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "RNN forward with aligned loads and __ldg optimization (CPU)");
}
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

#define NUM_STREAMS 4

// This function performs the exact same operation as the CUDA kernel code.
torch::Tensor forward(
    torch::Tensor x,
    std::vector<torch::Tensor> lstm_weights_ih,
    std::vector<torch::Tensor> lstm_weights_hh,
    std::vector<torch::Tensor> lstm_biases_ih,
    std::vector<torch::Tensor> lstm_biases_hh,
    torch::Tensor h0,
    torch::Tensor c0,
    bool is_training
) {
    // Ensure h0 and c0 are on the same device as x
    h0 = h0.to(x.device());
    c0 = c0.to(x.device());

    const int64_t num_layers = lstm_weights_ih.size();
    const int64_t batch_size = x.size(0);
    const int64_t seq_len = x.size(1);
    const int64_t input_size = x.size(2);
    const int64_t hidden_size = h0.size(2);

    // Determine chunk size for processing the sequence in parallel streams
    const int chunk_size = (seq_len + NUM_STREAMS - 1) / NUM_STREAMS;
    torch::Tensor out = x;

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        auto weight_ih = lstm_weights_ih[layer];
        auto weight_hh = lstm_weights_hh[layer];
        auto bias_ih = lstm_biases_ih[layer];
        auto bias_hh = lstm_biases_hh[layer];

        torch::Tensor h_layer = h0.select(0, layer);
        torch::Tensor c_layer = c0.select(0, layer);

        auto layer_out = torch::empty({batch_size, seq_len, hidden_size}, x.options());

        // Launch sequence chunks in separate CPU threads using OpenMP
        #pragma omp parallel for
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            for (int64_t seq_start = 0; seq_start < seq_len; seq_start += chunk_size) {
                int seq_chunk = std::min(chunk_size, static_cast<int>(seq_len - seq_start));
                if (seq_chunk <= 0) continue;

                const float* x_batch = x.data_ptr<float>() + batch * seq_len * input_size + seq_start * input_size;
                float* y_batch = layer_out.data_ptr<float>() + batch * seq_len * hidden_size + seq_start * hidden_size;
                float* h_ptr = h_layer.data_ptr<float>() + batch * hidden_size;
                float* c_ptr = c_layer.data_ptr<float>() + batch * hidden_size;

                const float* W_ih = weight_ih.data_ptr<float>();
                const float* W_hh = weight_hh.data_ptr<float>();
                const float* bias_ih_ptr = bias_ih.data_ptr<float>();
                const float* bias_hh_ptr = bias_hh.data_ptr<float>();

                for (int64_t tid = 0; tid < hidden_size; ++tid) {
                    float h_val = h_ptr[tid];
                    float c_val = c_ptr[tid];

                    for (int64_t t = seq_start; t < seq_start + seq_chunk; ++t) {
                        // Initialize gates with biases
                        float i_gate = bias_ih_ptr[tid] + bias_hh_ptr[tid];
                        float f_gate = bias_ih_ptr[hidden_size + tid] + bias_hh_ptr[hidden_size + tid];
                        float g_gate = bias_ih_ptr[2 * hidden_size + tid] + bias_hh_ptr[2 * hidden_size + tid];
                        float o_gate = bias_ih_ptr[3 * hidden_size + tid] + bias_hh_ptr[3 * hidden_size + tid];

                        // Input contribution
                        const float* x_t = x_batch + t * input_size;
                        for (int64_t k = 0; k < input_size; ++k) {
                            i_gate += x_t[k] * W_ih[(0 * hidden_size + tid) * input_size + k];
                            f_gate += x_t[k] * W_ih[(1 * hidden_size + tid) * input_size + k];
                            g_gate += x_t[k] * W_ih[(2 * hidden_size + tid) * input_size + k];
                            o_gate += x_t[k] * W_ih[(3 * hidden_size + tid) * input_size + k];
                        }

                        // Hidden state contribution from previous time step
                        for (int64_t k = 0; k < hidden_size; ++k) {
                            i_gate += h_ptr[k] * W_hh[(0 * hidden_size + tid) * hidden_size + k];
                            f_gate += h_ptr[k] * W_hh[(1 * hidden_size + tid) * hidden_size + k];
                            g_gate += h_ptr[k] * W_hh[(2 * hidden_size + tid) * hidden_size + k];
                            o_gate += h_ptr[k] * W_hh[(3 * hidden_size + tid) * hidden_size + k];
                        }

                        // Apply activation functions
                        i_gate = 1.0f / (1.0f + expf(-i_gate));
                        f_gate = 1.0f / (1.0f + expf(-f_gate));
                        g_gate = tanhf(g_gate);
                        o_gate = 1.0f / (1.0f + expf(-o_gate));

                        // Update cell state and hidden state
                        c_val = f_gate * c_val + i_gate * g_gate;
                        h_val = o_gate * tanhf(c_val);

                        // Write the output for this timestep
                        y_batch[t * hidden_size + tid] = h_val;
                    }

                    h_ptr[tid] = h_val;
                    c_ptr[tid] = c_val;
                }
            }
        }
    }

    return c0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LSTM forward with balanced workload distribution");
}
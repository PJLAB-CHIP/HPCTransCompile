#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace torch;

// Function to apply sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Function to apply tanh activation
inline float tanh_func(float x) {
    return tanhf(x);
}

// CPU version of the LSTM kernel
void lstm_balanced_workload_cpu(
    const float* x,           // [batch_size, total_seq_len, input_size]
    float* y,                 // [batch_size, total_seq_len, hidden_size]
    float* h,                 // [batch_size, hidden_size]
    float* c,                 // [batch_size, hidden_size]
    const float* W_ih,        // [4 * hidden_size, input_size]
    const float* W_hh,        // [4 * hidden_size, hidden_size]
    const float* bias_ih,     // [4 * hidden_size]
    const float* bias_hh,     // [4 * hidden_size]
    const int seq_start,       // Starting timestep index
    const int seq_length,      // Number of timesteps to process
    const int total_seq_len,   // Total number of timesteps
    const int input_size,
    const int hidden_size
) {
    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < 1; ++batch) { // Assuming batch_size = 1 for simplicity
        for (int tid = 0; tid < hidden_size; ++tid) {
            // Pointers for the batch
            const float* x_batch = x + batch * total_seq_len * input_size;
            float* y_batch = y + batch * total_seq_len * hidden_size;
            float* h_ptr = h + batch * hidden_size;
            float* c_ptr = c + batch * hidden_size;

            // Each thread loads its corresponding hidden and cell state
            float h_val = h_ptr[tid];
            float c_val = c_ptr[tid];

            // Process each timestep in the assigned chunk
            for (int t = seq_start; t < seq_start + seq_length; t++) {
                // Initialize gates with biases
                float i_gate = bias_ih[tid] + bias_hh[tid];
                float f_gate = bias_ih[hidden_size + tid] + bias_hh[hidden_size + tid];
                float g_gate = bias_ih[2 * hidden_size + tid] + bias_hh[2 * hidden_size + tid];
                float o_gate = bias_ih[3 * hidden_size + tid] + bias_hh[3 * hidden_size + tid];

                // Input contribution: process input vector
                const float* x_t = x_batch + t * input_size;
                for (int k = 0; k < input_size; k++) {
                    float x_val = x_t[k];
                    i_gate += x_val * W_ih[(0 * hidden_size + tid) * input_size + k];
                    f_gate += x_val * W_ih[(1 * hidden_size + tid) * input_size + k];
                    g_gate += x_val * W_ih[(2 * hidden_size + tid) * input_size + k];
                    o_gate += x_val * W_ih[(3 * hidden_size + tid) * input_size + k];
                }

                // Hidden state contribution from previous time step
                for (int k = 0; k < hidden_size; k++) {
                    float h_shared = h_ptr[k];
                    i_gate += h_shared * W_hh[(0 * hidden_size + tid) * hidden_size + k];
                    f_gate += h_shared * W_hh[(1 * hidden_size + tid) * hidden_size + k];
                    g_gate += h_shared * W_hh[(2 * hidden_size + tid) * hidden_size + k];
                    o_gate += h_shared * W_hh[(3 * hidden_size + tid) * hidden_size + k];
                }

                // Apply activation functions
                i_gate = sigmoid(i_gate);
                f_gate = sigmoid(f_gate);
                g_gate = tanh_func(g_gate);
                o_gate = sigmoid(o_gate);

                // Update cell state and hidden state
                c_val = f_gate * c_val + i_gate * g_gate;
                h_val = o_gate * tanh_func(c_val);

                // Write the output for this timestep
                y_batch[t * hidden_size + tid] = h_val;
            }

            // If this kernel processed the final chunk, write back the final states
            if (seq_start + seq_length == total_seq_len) {
                h_ptr[tid] = h_val;
                c_ptr[tid] = c_val;
            }
        }
    }
}

Tensor forward_cpu(
    Tensor x,
    std::vector<Tensor> lstm_weights_ih,
    std::vector<Tensor> lstm_weights_hh,
    std::vector<Tensor> lstm_biases_ih,
    std::vector<Tensor> lstm_biases_hh,
    Tensor h0,
    Tensor c0,
    bool is_training
) {
    const int64_t num_layers = lstm_weights_ih.size();
    const int64_t batch_size = x.size(0);
    const int64_t seq_len = x.size(1);
    const int64_t input_size = x.size(2);
    const int64_t hidden_size = h0.size(2);

    // Determine chunk size for processing the sequence
    const int chunk_size = (seq_len + 4 - 1) / 4; // Assuming 4 streams for simplicity
    Tensor out = x;

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        auto weight_ih = lstm_weights_ih[layer];
        auto weight_hh = lstm_weights_hh[layer];
        auto bias_ih = lstm_biases_ih[layer];
        auto bias_hh = lstm_biases_hh[layer];

        Tensor h_layer = h0.select(0, layer);
        Tensor c_layer = c0.select(0, layer);

        Tensor layer_out = torch::empty({batch_size, seq_len, hidden_size}, x.options());

        // Launch sequence chunks
        for (int stream_idx = 0; stream_idx < 4; stream_idx++) {
            int seq_start_idx = stream_idx * chunk_size;
            int seq_chunk = std::min(chunk_size, static_cast<int>(seq_len - seq_start_idx));
            if (seq_chunk <= 0) continue;
            lstm_balanced_workload_cpu(
                out.data_ptr<float>(),
                layer_out.data_ptr<float>(),
                h_layer.data_ptr<float>(),
                c_layer.data_ptr<float>(),
                weight_ih.data_ptr<float>(),
                weight_hh.data_ptr<float>(),
                bias_ih.data_ptr<float>(),
                bias_hh.data_ptr<float>(),
                seq_start_idx,
                seq_chunk,
                seq_len,
                (layer == 0) ? input_size : hidden_size,
                hidden_size
            );
        }

        out = layer_out;
    }

    return c0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "LSTM forward with balanced workload distribution on CPU");
}
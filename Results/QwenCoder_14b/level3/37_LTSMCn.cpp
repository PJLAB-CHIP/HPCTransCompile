#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace torch;

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

    Tensor out = x;

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        auto weight_ih = lstm_weights_ih[layer];
        auto weight_hh = lstm_weights_hh[layer];
        auto bias_ih = lstm_biases_ih[layer];
        auto bias_hh = lstm_biases_hh[layer];

        Tensor h_layer = h0.select(0, layer);
        Tensor c_layer = c0.select(0, layer);

        Tensor layer_out = torch::empty({batch_size, seq_len, hidden_size}, x.options());

        #pragma omp parallel for collapse(2)
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int tid = 0; tid < hidden_size; ++tid) {
                float h_val = h_layer[batch][tid].item<float>();
                float c_val = c_layer[batch][tid].item<float>();

                const float* x_batch = x[batch].data_ptr<float>();
                float* y_batch = layer_out[batch].data_ptr<float>();
                float* h_ptr = h_layer[batch].data_ptr<float>();
                float* c_ptr = c_layer[batch].data_ptr<float>();

                const float* W_ih = weight_ih.data_ptr<float>();
                const float* W_hh = weight_hh.data_ptr<float>();
                const float* bias_ih_data = bias_ih.data_ptr<float>();
                const float* bias_hh_data = bias_hh.data_ptr<float>();

                for (int t = 0; t < seq_len; ++t) {
                    float i_gate = bias_ih_data[tid] + bias_hh_data[tid];
                    float f_gate = bias_ih_data[hidden_size + tid] + bias_hh_data[hidden_size + tid];
                    float g_gate = bias_ih_data[2 * hidden_size + tid] + bias_hh_data[2 * hidden_size + tid];
                    float o_gate = bias_ih_data[3 * hidden_size + tid] + bias_hh_data[3 * hidden_size + tid];

                    const float* x_t = x_batch + t * input_size;
                    for (int k = 0; k < input_size; ++k) {
                        float x_val = x_t[k];
                        i_gate += x_val * W_ih[(0 * hidden_size + tid) * input_size + k];
                        f_gate += x_val * W_ih[(1 * hidden_size + tid) * input_size + k];
                        g_gate += x_val * W_ih[(2 * hidden_size + tid) * input_size + k];
                        o_gate += x_val * W_ih[(3 * hidden_size + tid) * input_size + k];
                    }

                    for (int k = 0; k < hidden_size; ++k) {
                        float h_shared = h_val;
                        i_gate += h_shared * W_hh[(0 * hidden_size + tid) * hidden_size + k];
                        f_gate += h_shared * W_hh[(1 * hidden_size + tid) * hidden_size + k];
                        g_gate += h_shared * W_hh[(2 * hidden_size + tid) * hidden_size + k];
                        o_gate += h_shared * W_hh[(3 * hidden_size + tid) * hidden_size + k];
                    }

                    i_gate = 1.0f / (1.0f + expf(-i_gate));
                    f_gate = 1.0f / (1.0f + expf(-f_gate));
                    g_gate = tanhf(g_gate);
                    o_gate = 1.0f / (1.0f + expf(-o_gate));

                    c_val = f_gate * c_val + i_gate * g_gate;
                    h_val = o_gate * tanhf(c_val);

                    y_batch[t * hidden_size + tid] = h_val;
                }

                h_ptr[tid] = h_val;
                c_ptr[tid] = c_val;
            }
        }

        out = layer_out;
    }

    return c0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "LSTM forward with balanced workload distribution");
}
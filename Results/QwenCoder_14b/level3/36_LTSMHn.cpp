#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <omp.h>

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
    auto out = x;
    auto hn = h0.clone();
    auto cn = c0.clone();

    const size_t num_layers = lstm_weights_ih.size();

    auto process_layer = [&](size_t i) {
        auto weight_ih = lstm_weights_ih[i];
        auto weight_hh = lstm_weights_hh[i];
        auto bias_ih = lstm_biases_ih[i];
        auto bias_hh = lstm_biases_hh[i];

        int64_t input_size = weight_ih.size(1);
        int64_t hidden_size = weight_hh.size(1);

        auto h_slice = hn.narrow(0, i, 1);
        auto c_slice = cn.narrow(0, i, 1);
        std::tuple<torch::Tensor, torch::Tensor> state_tuple = std::make_tuple(h_slice, c_slice);

        auto output_and_state = torch::lstm_cell(out, state_tuple, weight_ih, weight_hh, bias_ih, bias_hh);
        auto output = std::get<0>(output_and_state);
        auto state = std::get<1>(output_and_state);
        auto h_n = std::get<0>(state);
        auto c_n = std::get<1>(state);

        hn.narrow(0, i, 1).copy_(h_n);
        cn.narrow(0, i, 1).copy_(c_n);

        out = output;
    };

    #pragma omp parallel for
    for (size_t i = 0; i < num_layers; ++i) {
        process_layer(i);
    }

    return hn;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized LSTM forward (CPU)");
}
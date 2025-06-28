#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> 
lstm_module_fn(const torch::Tensor& input, 
              std::tuple<torch::Tensor, torch::Tensor> hx,
              const py::dict& params, 
              int64_t num_layers, 
              double dropout, 
              bool bidirectional) {
    
    // Initialize the weights vector
    std::vector<torch::Tensor> all_weights;
    
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        for (int64_t direction = 0; direction < (bidirectional ? 2 : 1); ++direction) {
            std::string suffix = (direction == 1) ? "_reverse" : "";
            
            // Get the parameters for this layer and direction
            std::string w_ih_key = "weight_ih_l" + std::to_string(layer) + suffix;
            std::string w_hh_key = "weight_hh_l" + std::to_string(layer) + suffix;
            std::string b_ih_key = "bias_ih_l" + std::to_string(layer) + suffix;
            std::string b_hh_key = "bias_hh_l" + std::to_string(layer) + suffix;
            
            torch::Tensor w_ih = params[py::str(w_ih_key)].cast<torch::Tensor>();
            torch::Tensor w_hh = params[py::str(w_hh_key)].cast<torch::Tensor>();
            torch::Tensor b_ih = params[py::str(b_ih_key)].cast<torch::Tensor>();
            torch::Tensor b_hh = params[py::str(b_hh_key)].cast<torch::Tensor>();
            
            // Append individual tensors to create a flat list
            all_weights.push_back(w_ih);
            all_weights.push_back(w_hh);
            all_weights.push_back(b_ih);
            all_weights.push_back(b_hh);
        }
    }
    
    // Extract hidden states
    torch::Tensor h0 = std::get<0>(hx);
    torch::Tensor c0 = std::get<1>(hx);
    
    // Call lstm with the properly formatted weights
    auto result = torch::lstm(input, 
                             {h0, c0}, 
                             all_weights, 
                             /*has_biases=*/true, 
                             num_layers, 
                             dropout, 
                             /*train=*/false, 
                             bidirectional, 
                             /*batch_first=*/true);
    
    torch::Tensor output = std::get<0>(result);
    torch::Tensor h_n = std::get<1>(result);
    torch::Tensor c_n = std::get<2>(result);
    
    return {output, {h_n, c_n}};
}

torch::Tensor linear_module_fn(const torch::Tensor& input, 
                              const torch::Tensor& weight, 
                              const torch::Tensor& bias) {
    return torch::linear(input, weight, bias);
}

torch::Tensor model_forward(const torch::Tensor& x, 
                       const torch::Tensor& h0, 
                       const torch::Tensor& c0,
                       const py::dict& params,
                       int64_t num_layers, 
                       double dropout, 
                       int64_t output_size, 
                       int64_t hidden_size, 
                       bool bidirectional = true) {
    
    py::dict lstm_params = params["lstm"].cast<py::dict>();
    py::dict fc_params = params["fc"].cast<py::dict>();
    
    auto lstm_result = lstm_module_fn(x, {h0, c0}, lstm_params, num_layers, dropout, bidirectional);
    
    torch::Tensor lstm_output = std::get<0>(lstm_result);
    
    // Get the last timestep output: lstm_output[:, -1, :]
    torch::Tensor last_output = lstm_output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
    
    // Apply linear layer
    torch::Tensor output = linear_module_fn(
        last_output, 
        fc_params["weight"].cast<torch::Tensor>(), 
        fc_params["bias"].cast<torch::Tensor>()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &model_forward, "lstm bidrectional forward");
}

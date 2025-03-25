#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Helper function to perform matrix multiplication
torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B) {
    auto result = torch::zeros({A.size(0), B.size(1)});
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A.size(0); ++i) {
        for (int j = 0; j < B.size(1); ++j) {
            for (int k = 0; k < A.size(1); ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Helper function to perform element-wise addition
torch::Tensor add(const torch::Tensor& A, const torch::Tensor& B) {
    auto result = torch::zeros_like(A);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A.size(0); ++i) {
        for (int j = 0; j < A.size(1); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

// Helper function to apply sigmoid activation
torch::Tensor sigmoid(const torch::Tensor& A) {
    auto result = torch::zeros_like(A);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A.size(0); ++i) {
        for (int j = 0; j < A.size(1); ++j) {
            result[i][j] = 1.0 / (1.0 + std::exp(-A[i][j].item<float>()));
        }
    }
    return result;
}

// Helper function to apply tanh activation
torch::Tensor tanh(const torch::Tensor& A) {
    auto result = torch::zeros_like(A);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A.size(0); ++i) {
        for (int j = 0; j < A.size(1); ++j) {
            result[i][j] = std::tanh(A[i][j].item<float>());
        }
    }
    return result;
}

torch::Tensor gru_forward(
    torch::Tensor x,
    std::vector<torch::Tensor> gru_weights_ih,
    std::vector<torch::Tensor> gru_weights_hh,
    std::vector<torch::Tensor> gru_biases_ih,
    std::vector<torch::Tensor> gru_biases_hh,
    torch::Tensor h0,
    bool is_training) {

    // Ensure h0 is on the CPU
    h0 = h0.to(torch::kCPU);

    // Prepare all_weights list matching PyTorch's expected format
    std::vector<torch::Tensor> all_weights;
    for (size_t i = 0; i < gru_weights_ih.size(); ++i) {
        // Ensure weights are on the CPU
        gru_weights_ih[i] = gru_weights_ih[i].to(torch::kCPU);
        gru_weights_hh[i] = gru_weights_hh[i].to(torch::kCPU);
        gru_biases_ih[i] = gru_biases_ih[i].to(torch::kCPU);
        gru_biases_hh[i] = gru_biases_hh[i].to(torch::kCPU);
        
        all_weights.push_back(gru_weights_ih[i]);
        all_weights.push_back(gru_weights_hh[i]);
        all_weights.push_back(gru_biases_ih[i]);
        all_weights.push_back(gru_biases_hh[i]);
    }

    // Calculate num_layers from bidirectional setup
    int num_layers = gru_weights_ih.size() / 2;

    // Initialize hidden state
    torch::Tensor h = h0;

    // Perform GRU forward pass
    for (int layer = 0; layer < num_layers; ++layer) {
        auto W_ir = all_weights[layer * 4];
        auto W_iz = all_weights[layer * 4 + 1];
        auto W_in = all_weights[layer * 4 + 2];
        auto b_ir = all_weights[layer * 4 + 3];

        auto W_hr = all_weights[layer * 4 + 4];
        auto W_hz = all_weights[layer * 4 + 5];
        auto W_hn = all_weights[layer * 4 + 6];
        auto b_hz = all_weights[layer * 4 + 7];

        auto r_t = sigmoid(add(matmul(h, W_hr), add(matmul(x, W_ir), b_ir)));
        auto z_t = sigmoid(add(matmul(h, W_hz), add(matmul(x, W_iz), b_hz)));
        auto n_tilde = tanh(add(matmul(h, W_hn), add(matmul(x, W_in), b_in)));

        h = add(mul(r_t, h), mul((1 - z_t), n_tilde));
    }

    return h;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gru_forward, "GRU forward (CPU)");
}
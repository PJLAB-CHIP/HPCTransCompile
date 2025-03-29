#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Function to compute the sum of squares
float compute_sum_of_squares(const float* input, int numel) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < numel; ++i) {
        sum += input[i] * input[i];
    }
    return sum;
}

// Function to normalize the tensor
void normalize_tensor(const float* input, float* output, float norm, int numel) {
    #pragma omp parallel for
    for (int i = 0; i < numel; ++i) {
        output[i] = input[i] / norm;
    }
}

// C++ forward function called from Python
torch::Tensor forward(torch::Tensor input) {
    // Validate input constraints
    TORCH_CHECK(!input.is_cuda(), "Input tensor must be on CPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Allocate output tensor and a variable for the norm
    auto output = torch::empty_like(input);
    float norm_val;

    // Raw pointers
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    int numel = input.numel();

    // Compute sum of squares
    norm_val = compute_sum_of_squares(input_ptr, numel);

    // Compute the square root of the sum of squares
    norm_val = std::sqrt(norm_val);

    // Normalize the tensor
    normalize_tensor(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular Frobenius norm normalization");
}

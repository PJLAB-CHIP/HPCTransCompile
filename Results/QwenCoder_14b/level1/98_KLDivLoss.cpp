#include <torch/extension.h>
#include <omp.h>
#include <cmath>

// Function to perform reduction using OpenMP
float omp_reduce(const float* data, int size) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

// CPU implementation of the first stage of KL divergence calculation
void kl_div_cpu_stage1(
    const float* log_predictions,
    const float* targets,
    float* block_results,
    const int n) {
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        block_results[i] = std::exp(log_pred) - target * log_pred;
    }
}

// CPU implementation of the second stage of KL divergence calculation
void kl_div_cpu_stage2(
    const float* block_results,
    float* output,
    const int num_blocks,
    const float normalizer) {
    
    float sum = omp_reduce(block_results, num_blocks);
    output[0] = sum * normalizer;
}

torch::Tensor kl_div_cpu_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = std::min((n + threads * 8 - 1) / (threads * 8), 1024);
    const float normalizer = 1.0f / static_cast<float>(n);
    
    auto block_results = torch::empty({blocks}, log_predictions.options());
    
    kl_div_cpu_stage1(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );
    
    kl_div_cpu_stage2(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks,
        normalizer
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cpu_forward, "KL divergence forward (CPU)");
}

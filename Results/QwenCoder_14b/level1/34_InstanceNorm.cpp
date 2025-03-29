#include <torch/extension.h>
#include <omp.h>
#include <cmath>

// Function to perform reduction sum using OpenMP
float omp_reduce_sum(const float* data, int size) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

// Function to perform reduction sum of squares using OpenMP
float omp_reduce_sum_sq(const float* data, int size) {
    float sum_sq = 0.0f;
    #pragma omp parallel for reduction(+:sum_sq)
    for (int i = 0; i < size; ++i) {
        sum_sq += data[i] * data[i];
    }
    return sum_sq;
}

// CPU implementation of instance normalization
void instance_norm_cpu(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int N,
    int C,
    int H,
    int W,
    float eps
) {
    int HW = H * W;
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* x_ptr = x + (n * C + c) * HW;
            float* y_ptr = y + (n * C + c) * HW;
            
            // Compute partial sums
            float sum = omp_reduce_sum(x_ptr, HW);
            float sum_sq = omp_reduce_sum_sq(x_ptr, HW);
            
            // Compute mean and variance
            float mean = sum / HW;
            float var = (sum_sq / HW) - (mean * mean);
            var = (var < 0.f) ? 0.f : var;
            float invstd = std::sqrt(1.0f / (var + eps));
            
            // Load scale and bias once per thread if they exist
            float scale = weight ? weight[c] : 1.0f;
            float shift = bias ? bias[c] : 0.0f;
            
            // Normalize
            #pragma omp parallel for
            for (int i = 0; i < HW; ++i) {
                float val = x_ptr[i];
                val = (val - mean) * invstd;
                y_ptr[i] = val * scale + shift;
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor");
    if (weight.defined()) TORCH_CHECK(!weight.is_cuda(), "weight must be a CPU tensor");
    if (bias.defined()) TORCH_CHECK(!bias.is_cuda(), "bias must be a CPU tensor");
    
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");
    
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    
    auto y = torch::empty_like(x);
    
    instance_norm_cpu(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CPU)");
}

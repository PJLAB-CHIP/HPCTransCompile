#include <torch/extension.h>
#include <math.h>
#include <omp.h>

// Fused kernel: concatenates x and hidden and computes the linear transform in one pass.
// Each block (one warp of 32 threads) computes one dot product for one (row, output) pair.

// Kernel parameters:
// x: [batch_size, x_size]
// hidden: [batch_size, hidden_size]
// i2h_weight: [out_size, total_width] where total_width = x_size + hidden_size
// i2h_bias: [out_size]
// hidden_new_out: [batch_size, out_size] output after tanh( i2h_bias + dot )
// batch_size, x_size, hidden_size, out_size are dimensions

void fused_concat_linear_kernel_cpu(
    const float* __restrict__ x,
    const float* __restrict__ hidden,
    const float* __restrict__ i2h_weight,
    const float* __restrict__ i2h_bias,
    float* __restrict__ hidden_new_out,
    const int batch_size,
    const int x_size,
    const int hidden_size,
    const int out_size,
    const int total_dot_products,
    const int threads
) {
    int total_width = x_size + hidden_size;

    #pragma omp parallel for schedule(dynamic)
    for (int global_idx = 0; global_idx < total_dot_products; ++global_idx) {
        int row = global_idx / out_size;
        int out_idx = global_idx % out_size;

        if (row >= batch_size) continue;

        float sum = 0.0f;
        for (int k = 0; k < total_width; k += threads) {
            int lane = k + omp_get_thread_num();
            if (lane < total_width) {
                float a = (lane < x_size) ? x[row * x_size + lane] : hidden[row * hidden_size + (lane - x_size)];
                float b = i2h_weight[out_idx * total_width + lane];
                sum += a * b;
            }
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            sum += sum;
        }

        if (omp_get_thread_num() == 0) {
            float result = tanhf(sum + i2h_bias[out_idx]);
            hidden_new_out[row * out_size + out_idx] = result;
        }
    }
}

// Host function
// This fused kernel replaces the separate concatenation and addmm (i2h) operations.
// It computes hidden_new = tanh(i2h_bias + [x, hidden] * i2h_weight^T) in one pass,
// avoiding the allocation and memory traffic of an intermediate concatenated tensor.

torch::Tensor module_fn_cpu(
    torch::Tensor x,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias,
    torch::Tensor hidden
) {
    // Ensure tensors are contiguous
    x = x.contiguous();
    i2h_weight = i2h_weight.contiguous();
    i2h_bias = i2h_bias.contiguous();
    h2o_weight = h2o_weight.contiguous();
    h2o_bias = h2o_bias.contiguous();
    hidden = hidden.contiguous();

    const int batch_size = x.size(0);
    const int x_size = x.size(1);
    const int hidden_size = hidden.size(1);
    // out_size is the number of neurons in the i2h linear transform (i2h_bias length)
    const int out_size = i2h_bias.size(0);
    int total_width = x_size + hidden_size;

    // Allocate tensor for hidden_new output of fused i2h operation
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor hidden_new = torch::empty({batch_size, out_size}, options);

    int total_dot_products = batch_size * out_size;
    int threads = omp_get_max_threads();

    fused_concat_linear_kernel_cpu(
        x.data_ptr<float>(),
        hidden.data_ptr<float>(),
        i2h_weight.data_ptr<float>(),
        i2h_bias.data_ptr<float>(),
        hidden_new.data_ptr<float>(),
        batch_size,
        x_size,
        hidden_size,
        out_size,
        total_dot_products,
        threads
    );

    // Compute the final output: h2o_bias + hidden_new * h2o_weight^T
    // This step is kept separate and uses optimized torch::addmm
    torch::Tensor output = torch::addmm(h2o_bias, hidden_new, h2o_weight.t());
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu, "Fused Module forward (CPU) using warp-level primitives");
}
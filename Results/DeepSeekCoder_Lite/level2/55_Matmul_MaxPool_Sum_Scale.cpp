#include <torch/extension.h>
#include <omp.h>

// Optimized kernel using shared memory for input vector and intermediate results.
// One block per batch element; multiple threads per block cooperate to compute the linear transform and pooling.

void module_fn_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,           // batch_size
    int inF,         // number of input features
    int outF,        // number of output features
    int kernel_size,
    float scale_factor,
    int block_size
) {
    #pragma omp parallel for
    for (int b = 0; b < B; ++b) {
        // Dynamically allocated shared memory layout:
        // - First: shared input vector (size inF)
        // - Second: intermediate output vector for F.linear (size outF)
        // - Third: partial sums for pooling reduction (size = blockDim.x)
        float* shared_x = new float[inF];
        float* local_out = new float[outF];
        float* partial_sums = new float[block_size];

        // Load x[b, :] into shared memory
        for (int i = 0; i < inF; ++i) {
            shared_x[i] = x[b * inF + i];
        }

        // Compute F.linear: each thread computes dot-product(s) for assigned output indices
        for (int j = 0; j < outF; ++j) {
            float temp = bias[j];
            for (int i = 0; i < inF; ++i) {
                temp += shared_x[i] * weight[j * inF + i];
            }
            local_out[j] = temp;
        }

        // Max pooling along the outF dimension with window size 'kernel_size'
        int pooled_len = 1 + (outF - kernel_size) / kernel_size;

        // Each thread processes a subset of pooling segments and accumulates their max values
        float sum_local = 0.0f;
        for (int seg = 0; seg < pooled_len; ++seg) {
            int start = seg * kernel_size;
            float m_val = local_out[start];
            for (int offset = 1; offset < kernel_size; ++offset) {
                float curr = local_out[start + offset];
                m_val = (curr > m_val) ? curr : m_val;
            }
            sum_local += m_val;
        }
        partial_sums[0] = sum_local;

        // Parallel reduction to sum partial pooling results
        for (int stride = block_size / 2; stride > 0; stride /= 2) {
            partial_sums[0] += partial_sums[stride];
        }

        // Apply scaling and write result
        output[b] = partial_sums[0] * scale_factor;

        // Free allocated memory
        delete[] shared_x;
        delete[] local_out;
        delete[] partial_sums;
    }
}

// Forward function callable from Python
at::Tensor forward(
    at::Tensor x,
    int64_t kernel_size,
    double scale_factor,
    at::Tensor weight,
    at::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must have shape (batch_size, in_features)");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape (out_features, in_features)");
    TORCH_CHECK(bias.dim() == 1, "bias must have shape (out_features)");

    const auto B = x.size(0);
    const auto inF = x.size(1);
    const auto outF = weight.size(0);
    
    auto out = torch::empty({B}, x.options());

    // Use 256 threads per block for parallel processing within each batch sample
    const int block_size = 256;

    // Compute required shared memory: for shared_x (inF), local_out (outF), and partial_sums (threads)
    size_t sharedMemSize = (inF + outF + block_size) * sizeof(float);

    module_fn_kernel(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        inF,
        outF,
        (int)kernel_size,
        (float)scale_factor,
        block_size
    );

    return out;
}

// Pybind11 module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward for module_fn with shared memory optimization");
}
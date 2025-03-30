#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

#define ELEMENTS_PER_THREAD 4
#define SHARED_MEM_SIZE (BLOCK_SIZE * ELEMENTS_PER_THREAD)

__forceinline__ float fused_mish_tanh_activation(float x) {
    float softplus = logf(1.0f + expf(x));
    float mish = x * tanhf(softplus);
    return tanhf(mish);
}

void shared_mem_mish_tanh_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int total_elements,
    const int BLOCK_SIZE
) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int elements_per_thread = ELEMENTS_PER_THREAD;
        int shared_mem_size = BLOCK_SIZE * elements_per_thread;

        // Allocate shared memory
        __shared__ float shared_data[SHARED_MEM_SIZE];

        // Load data into shared memory in a coalesced manner
        for (int i = tid; i < total_elements; i += num_threads) {
            int idx = i / elements_per_thread;
            shared_data[tid + i % elements_per_thread * BLOCK_SIZE] = input[idx];
        }

        // Synchronize to ensure all threads have loaded data
        #pragma omp barrier

        // Process data in shared memory
        for (int i = tid; i < total_elements; i += num_threads) {
            int idx = i / elements_per_thread;
            float val = shared_data[tid + i % elements_per_thread * BLOCK_SIZE];
            output[idx] = fused_mish_tanh_activation(val);
        }
    }
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias
) {
    TORCH_CHECK(x.is_cpu(), "Input tensor x must be a CPU tensor");
    TORCH_CHECK(conv_weight.is_cpu(), "Convolution weight must be a CPU tensor");
    TORCH_CHECK(conv_bias.is_cpu(), "Convolution bias must be a CPU tensor");

    auto x_conv = at::conv3d(
        x,
        conv_weight,
        conv_bias,
        {stride, stride, stride},
        {padding, padding, padding}
    );

    auto output = torch::empty_like(x_conv);
    const int total_elements = x_conv.numel();

    // Calculate the number of blocks needed
    int num_blocks = (total_elements + SHARED_MEM_SIZE - 1) / SHARED_MEM_SIZE;
    int BLOCK_SIZE = 256; // Assuming BLOCK_SIZE is defined somewhere

    shared_mem_mish_tanh_kernel(
        output.data_ptr<float>(),
        x_conv.data_ptr<float>(),
        total_elements,
        BLOCK_SIZE
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Shared memory optimized convolution with Mish and Tanh activations (CPU)");
}
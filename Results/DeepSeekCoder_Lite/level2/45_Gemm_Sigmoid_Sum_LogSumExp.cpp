#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Device function for computing dot product
__forceinline__ float dot_product(
    const float* __restrict__ vec1,
    const float* __restrict__ vec2,
    const int size
) {
    float result = 0.0f;
    for (int i = 0; i < size; ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Device function for sigmoid activation
__forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device function for parallel reduction in shared memory
__forceinline__ float block_reduce_sum(float val, float* shared, const int tid) {
    shared[tid] = val;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    return shared[0];
}

// Device function for parallel max reduction
__forceinline__ float block_reduce_max(float val, float* shared, const int tid) {
    shared[tid] = val;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    return shared[0];
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = weight.size(0);

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());

    auto final_output = torch::empty({1}, options);

    #pragma omp parallel for
    for (int row = 0; row < batch_size; ++row) {
        const float* row_input = &input[row * input_size];
        float local_sum = 0.0f;

        for (int col = 0; col < hidden_size; ++col) {
            const float* col_weight = &weight[col * input_size];
            float dot = dot_product(row_input, col_weight, input_size);
            dot += bias[col];
            local_sum += sigmoid(dot);
        }

        float row_total = 0.0f;
        #pragma omp critical
        {
            row_total = local_sum;
        }

        float local_max = -INFINITY;
        for (int i = 0; i < batch_size; ++i) {
            local_max = max(local_max, output[i]);
        }

        float max_val = local_max;
        float sum_exp_val = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum_exp_val += exp(output[i] - max_val);
        }

        final_output[0] = log(sum_exp_val) + max_val;
    }

    return final_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward pass");
}
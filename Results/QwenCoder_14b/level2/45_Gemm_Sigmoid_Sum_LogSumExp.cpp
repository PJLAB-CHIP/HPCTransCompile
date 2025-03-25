#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Function for computing dot product
float dot_product(const float* vec1, const float* vec2, const int size) {
    float result = 0.0f;
    for (int i = 0; i < size; ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Function for sigmoid activation
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Function for parallel reduction in shared memory
float block_reduce_sum(float val, float* shared, const int tid) {
    shared[tid] = val;
    #pragma omp barrier

    #pragma omp single
    {
        for (unsigned int s = omp_get_num_threads() / 2; s > 0; s >>= 1) {
            for (int i = 0; i < s; ++i) {
                shared[i] += shared[i + s];
            }
        }
    }
    #pragma omp barrier
    return shared[0];
}

// Function for parallel max reduction
float block_reduce_max(float val, float* shared, const int tid) {
    shared[tid] = val;
    #pragma omp barrier

    #pragma omp single
    {
        for (unsigned int s = omp_get_num_threads() / 2; s > 0; s >>= 1) {
            for (int i = 0; i < s; ++i) {
                shared[i] = max(shared[i], shared[i + s]);
            }
        }
    }
    #pragma omp barrier
    return shared[0];
}

void fused_gemm_sigmoid_logsumexp_cpu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int input_size,
    const int hidden_size
) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        float local_sum = 0.0f;
        float shared_mem[num_threads];

        #pragma omp for
        for (int row = 0; row < batch_size; ++row) {
            const float* row_input = &input[row * input_size];

            for (int col = tid; col < hidden_size; col += num_threads) {
                const float* col_weight = &weight[col * input_size];
                float dot = dot_product(row_input, col_weight, input_size);
                dot += bias[col];
                local_sum += sigmoid(dot);
            }

            float row_total = block_reduce_sum(local_sum, shared_mem, tid);
            if (tid == 0) {
                output[row] = row_total;
            }
        }

        #pragma omp barrier

        float local_max = -INFINITY;
        #pragma omp for
        for (int i = tid; i < batch_size; i += num_threads) {
            local_max = max(local_max, output[i]);
        }
        float max_val = block_reduce_max(local_max, shared_mem, tid);
        #pragma omp barrier

        float local_exp_sum = 0.0f;
        #pragma omp for
        for (int i = tid; i < batch_size; i += num_threads) {
            local_exp_sum += expf(output[i] - max_val);
        }
        float sum_exp_val = block_reduce_sum(local_exp_sum, shared_mem, tid);

        if (tid == 0) {
            output[0] = logf(sum_exp_val) + max_val;
        }
    }
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

    fused_gemm_sigmoid_logsumexp_cpu(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return final_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward pass");
}
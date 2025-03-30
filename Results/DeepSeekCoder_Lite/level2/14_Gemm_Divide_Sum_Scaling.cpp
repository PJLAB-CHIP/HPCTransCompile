#include <torch/extension.h>
#include <omp.h>

template <int BLOCK_THREADS>
void custom_kernel(
    const float *x,
    const float *weight,
    float *output,
    float scaling_factor,
    int input_size,
    int hidden_size,
    int batch_size,
    int thread_idx) {

    float *x_shared = new float[input_size];
    int batch_idx = thread_idx / BLOCK_THREADS;
    if (batch_idx >= batch_size) return;

    int tid = thread_idx % BLOCK_THREADS;

    // Load current batch into shared memory
    for (int k = tid; k < input_size; k += BLOCK_THREADS) {
        x_shared[k] = x[batch_idx * input_size + k];
    }

    float thread_sum = 0.0f;
    int j_per_thread = (hidden_size + BLOCK_THREADS - 1) / BLOCK_THREADS;
    int start_j = tid * j_per_thread;
    int end_j = min((tid + 1) * j_per_thread, hidden_size);

    for (int j = start_j; j < end_j; ++j) {
        const float *weight_row = weight + j * input_size;
        float dot = 0.0f;
        for (int k = 0; k < input_size; ++k) {
            dot += x_shared[k] * weight_row[k];
        }
        thread_sum += dot;
    }

    // Block-wide reduction
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            thread_sum += x_shared[tid + stride];
        }
        __syncwarp();
    }

    if (tid == 0) {
        output[batch_idx] = (thread_sum / 2.0f) * scaling_factor;
    }

    delete[] x_shared;
}

torch::Tensor forward_cpu(
    torch::Tensor x,
    float scaling_factor,
    torch::Tensor weight) {

    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::zeros({batch_size, 1}, x.options());

    const int BLOCK_THREADS = 256;
    int num_threads = batch_size * BLOCK_THREADS;

    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        custom_kernel<BLOCK_THREADS>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            scaling_factor,
            input_size,
            hidden_size,
            batch_size,
            thread_idx);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Custom forward CPU function");
}
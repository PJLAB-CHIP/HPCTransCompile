#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int BLOCK_THREADS>
__global__ void custom_kernel(
    const float *x,
    const float *weight,
    float *output,
    float scaling_factor,
    int input_size,
    int hidden_size,
    int batch_size) {

    extern __shared__ float x_shared[];
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;

    // Load current batch into shared memory
    for (int k = tid; k < input_size; k += BLOCK_THREADS) {
        x_shared[k] = x[batch_idx * input_size + k];
    }
    __syncthreads();

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
    __shared__ float shared_sum[BLOCK_THREADS];
    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = (shared_sum[0] / 2.0f) * scaling_factor;
    }
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    float scaling_factor,
    torch::Tensor weight) {

    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::zeros({batch_size, 1}, x.options());

    const int BLOCK_THREADS = 256;
    dim3 grid(batch_size);
    dim3 block(BLOCK_THREADS);
    size_t shared_mem = input_size * sizeof(float);

    custom_kernel<BLOCK_THREADS><<<grid, block, shared_mem>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        input_size,
        hidden_size,
        batch_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Custom forward CUDA function");
}
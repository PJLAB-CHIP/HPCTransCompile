#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the number of threads per block used for intra-block reduction
constexpr int THREADS_PER_BLOCK = 128;

// Optimized kernel with ReLU activation using shared memory and warp-level primitives
// Minimized warp divergence by ensuring uniform control flow
template <typename T>
__global__ void optimized_linear_relu_kernel_uniform(
    const T* __restrict__ input,
    int in_dim,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output) {

    // Each block computes one output element
    int batch = blockIdx.x;      // index of the input sample
    int j = blockIdx.y;          // output neuron index
    int tid = threadIdx.x;
    T sum = (T)0;

    // Each thread accumulates partial dot product from the input vector and the weight row
    int t = tid;
    int limit = in_dim - (blockDim.x * 3);
    for (; t < limit; t += 4 * blockDim.x) {
        sum += input[batch * in_dim + t] * weight[j * in_dim + t] +
               input[batch * in_dim + t + blockDim.x] * weight[j * in_dim + t + blockDim.x] +
               input[batch * in_dim + t + 2 * blockDim.x] * weight[j * in_dim + t + 2 * blockDim.x] +
               input[batch * in_dim + t + 3 * blockDim.x] * weight[j * in_dim + t + 3 * blockDim.x];
    }
    for (; t < in_dim; t += blockDim.x) {
        sum += input[batch * in_dim + t] * weight[j * in_dim + t];
    }

    // Warp-level reduction using shuffle
    unsigned int mask = 0xffffffff;
    int lane = tid & 31; // equivalent to tid % warpSize
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Use shared memory to store one reduced sum per warp
    extern __shared__ T sdata[]; // shared memory size provided at kernel launch
    int warpId = tid / 32;
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();

    // Final reduction: first warp reduces the per-warp sums
    T blockSum = (T)0;
    int numWarps = blockDim.x / 32;
    if (tid < numWarps) {
        blockSum = sdata[tid];
        for (int offset = numWarps / 2; offset > 0; offset /= 2) {
            blockSum += __shfl_down_sync(mask, blockSum, offset);
        }
        // Uniform control flow: all threads in the warp execute the same instructions
        T z = blockSum + bias[j];
        output[batch * gridDim.y + j] = z > (T)0 ? z : (T)0; // ReLU activation
    }
}

// Optimized kernel without activation using shared memory and warp-level primitives
// Minimized warp divergence by ensuring uniform control flow
template <typename T>
__global__ void optimized_linear_kernel_uniform(
    const T* __restrict__ input,
    int in_dim,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output) {

    int batch = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;
    T sum = (T)0;
    int t = tid;
    int limit = in_dim - (blockDim.x * 3);
    for (; t < limit; t += 4 * blockDim.x) {
        sum += input[batch * in_dim + t] * weight[j * in_dim + t] +
               input[batch * in_dim + t + blockDim.x] * weight[j * in_dim + t + blockDim.x] +
               input[batch * in_dim + t + 2 * blockDim.x] * weight[j * in_dim + t + 2 * blockDim.x] +
               input[batch * in_dim + t + 3 * blockDim.x] * weight[j * in_dim + t + 3 * blockDim.x];
    }
    for (; t < in_dim; t += blockDim.x) {
        sum += input[batch * in_dim + t] * weight[j * in_dim + t];
    }

    unsigned int mask = 0xffffffff;
    int lane = tid & 31;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    extern __shared__ T sdata[];
    int warpId = tid / 32;
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();

    T blockSum = (T)0;
    int numWarps = blockDim.x / 32;
    if (tid < numWarps) {
        blockSum = sdata[tid];
        for (int offset = numWarps / 2; offset > 0; offset /= 2) {
            blockSum += __shfl_down_sync(mask, blockSum, offset);
        }
        // Uniform control flow: all threads in the warp execute the same instructions
        output[batch * gridDim.y + j] = blockSum + bias[j];
    }
}

// The forward function iterates through layers of the MLP
torch::Tensor forward(
    torch::Tensor x,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases) {

    TORCH_CHECK(weights.size() == biases.size(), "Weights and biases count mismatch");
    TORCH_CHECK(x.size(1) == weights[0].size(1), "Input dimension mismatch");

    torch::Tensor current_input = x;

    // Process all layers except the last one with ReLU activation
    for (size_t i = 0; i < weights.size() - 1; i++) {
        auto weight = weights[i];
        auto bias = biases[i];
        int in_dim = weight.size(1);
        int out_dim = weight.size(0);
        int batch_size = current_input.size(0);

        auto output = torch::zeros({batch_size, out_dim}, 
            torch::device(torch::kCUDA).dtype(current_input.dtype()));

        // Launch configuration: one block per output element
        dim3 grid(batch_size, out_dim);
        dim3 block(THREADS_PER_BLOCK);
        size_t shared_mem = (THREADS_PER_BLOCK / 32) * sizeof(float);

        if (current_input.dtype() == torch::kFloat32) {
            optimized_linear_relu_kernel_uniform<float><<<grid, block, shared_mem>>>(
                current_input.data_ptr<float>(),
                in_dim,
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                output.data_ptr<float>());
        } else {
            TORCH_CHECK(false, "Unsupported dtype");
        }
        current_input = output;
    }

    // Last layer without ReLU activation
    auto weight = weights.back();
    auto bias = biases.back();
    int in_dim = weight.size(1);
    int out_dim = weight.size(0);
    int batch_size = current_input.size(0);

    auto output = torch::zeros({batch_size, out_dim}, 
        torch::device(torch::kCUDA).dtype(current_input.dtype()));

    dim3 grid(batch_size, out_dim);
    dim3 block(THREADS_PER_BLOCK);
    size_t shared_mem = (THREADS_PER_BLOCK / 32) * sizeof(float);

    if (current_input.dtype() == torch::kFloat32) {
        optimized_linear_kernel_uniform<float><<<grid, block, shared_mem>>>(
            current_input.data_ptr<float>(),
            in_dim,
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>());
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized MLP forward (CUDA)");
}

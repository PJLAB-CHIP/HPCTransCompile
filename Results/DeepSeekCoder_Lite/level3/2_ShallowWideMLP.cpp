#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Reserve 64KB constant memory for biases
__constant__ char constant_bias[65536];

// Optimized MLP kernel with conditional synchronization
// Each block computes one output element using warp-level reduction
// __syncthreads() is called only when blockDim.x > warpSize

template <typename scalar_t>
void mlp_forward_kernel_min_sync(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const int num_threads) {

    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int row = 0; row < batch_size; row++) {
        for (int col = 0; col < out_features; col++) {
            scalar_t sum = 0;
            const int input_offset = row * in_features;
            const int weight_offset = col * in_features;

            // Loop over in_features with 4x unrolling
            const int total = in_features;
            const int stride = 256 * 4;
            for (int i = 0; i < total; i += stride) {
                scalar_t temp = 0;
                if (i + 3 < total) {
                    temp = input[input_offset + i]     * weight[weight_offset + i] +
                           input[input_offset + i + 1] * weight[weight_offset + i + 1] +
                           input[input_offset + i + 2] * weight[weight_offset + i + 2] +
                           input[input_offset + i + 3] * weight[weight_offset + i + 3];
                } else {
                    for (int j = 0; j < 4 && (i + j) < total; j++) {
                        temp += input[input_offset + i + j] * weight[weight_offset + i + j];
                    }
                }
                sum += temp;
            }

            // Warp-level reduction using shuffle intrinsics
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __builtin_shuffle_float(sum, sum, offset);
            }

            // Final reduction by thread 0
            if (omp_get_thread_num() == 0) {
                scalar_t final_sum = 0;
                for (int w = 0; w < num_threads; w++) {
                    final_sum += sum;
                }
                // Add bias from constant memory
                final_sum += reinterpret_cast<const scalar_t*>(constant_bias)[col];
                output[row * out_features + col] = final_sum;
            }
        }
    }
}

// Standard ReLU kernel remains unchanged

template <typename scalar_t>
void relu_kernel(
    scalar_t* __restrict__ data,
    const int size,
    const int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

// Host function chaining layers: launching the MLP kernel and applying ReLU between layers

torch::Tensor mlp_cuda_forward(
    torch::Tensor input,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases) {

    auto device = input.device();
    int num_layers = weights.size();
    auto current = input;

    for (int i = 0; i < num_layers; i++) {
        int batch_size = current.size(0);
        int in_features = current.size(1);
        int out_features = weights[i].size(0);

        auto output = torch::empty({batch_size, out_features}, 
                                     torch::dtype(current.dtype()).device(device));

        // Copy bias to constant memory if it fits
        AT_DISPATCH_FLOATING_TYPES(current.scalar_type(), "bias_copy", ([&] {
            size_t bias_bytes = out_features * sizeof(scalar_t);
            if (bias_bytes <= sizeof(constant_bias)) {
                cudaMemcpyToSymbol(constant_bias, biases[i].data_ptr<scalar_t>(), bias_bytes);
            }
        }));

        // Configure thread block size and dynamic shared memory size
        const int threads = 256;  // Must be a multiple of warpSize
        int shared_mem_size = (threads / 32) * sizeof(float);  // Allocation for warp-level sums
        dim3 blocks(batch_size, out_features);

        AT_DISPATCH_FLOATING_TYPES(current.scalar_type(), "mlp_forward_kernel_min_sync", ([&] {
            mlp_forward_kernel_min_sync<scalar_t>(
                current.data_ptr<scalar_t>(),
                weights[i].data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                in_features,
                out_features,
                omp_get_max_threads()
            );
        }));

        // Apply ReLU activation for intermediate layers
        if (i < num_layers - 1) {
            int size = batch_size * out_features;
            int thread_per_block = 256;
            int num_blocks = (size + thread_per_block - 1) / thread_per_block;
            AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "relu_kernel", ([&] {
                relu_kernel<scalar_t>(
                    output.data_ptr<scalar_t>(),
                    size,
                    omp_get_max_threads()
                );
            }));
        }

        current = output;
    }

    return current;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mlp_cuda_forward, "MLP forward (CUDA) with minimized synchronizations");
}
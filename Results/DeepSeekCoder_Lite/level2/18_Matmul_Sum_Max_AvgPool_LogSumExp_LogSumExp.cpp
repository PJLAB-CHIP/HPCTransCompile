#include <torch/extension.h>
#include <omp.h>

#define WARP_SIZE 32

template <typename scalar_t>
void warp_atomic_sequence_ops_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const int batch_idx,
    const int lane,
    const int warp_id,
    const int warps_per_block) {

    // Initialize the output for this batch element exactly once
    if (lane == 0 && warp_id == 0) {
        output[batch_idx] = 0;
    }

    // Each warp will accumulate its partial result in register
    scalar_t warp_partial = 0;

    // Distribute output neurons among warps: each warp processes neurons starting from its warp_id,
    // stepping by warps_per_block
    for (int o = warp_id; o < out_features; o += warps_per_block) {
        scalar_t sum_o = 0;
        // Each thread in the warp processes a portion of the dot product over in_features
        for (int i = lane; i < in_features; i += WARP_SIZE) {
            sum_o += x[batch_idx * in_features + i] * weight[o * in_features + i];
        }
        // Reduce the partial dot product within the warp using warp shuffle
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum_o += __shfl_down_sync(0xffffffff, sum_o, offset);
        }
        // Lane 0 of the warp now has the complete dot product for output neuron o, add bias and accumulate
        if (lane == 0) {
            warp_partial += (bias[o] + sum_o);
        }
    }

    // Use atomicAdd from each warp's leader to accumulate the final sum for the batch element
    if (lane == 0) {
        output[batch_idx] += warp_partial;
    }
}

// Host function to launch the kernel
torch::Tensor sequence_ops_cpu_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, 1}, x.options());

    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int threads_x = WARP_SIZE; // must be 32
        const int warps_per_block = 8;     // can be tuned based on problem size
        const int threads_y = warps_per_block;
        const dim3 threads(threads_x, threads_y);
        const dim3 blocks(1);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "warp_atomic_sequence_ops_cpu", ([&] {
            warp_atomic_sequence_ops_kernel<scalar_t>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                in_features,
                out_features,
                batch_idx,
                0, // lane
                0, // warp_id
                8  // warps_per_block
            );
        }));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sequence_ops_cpu_forward, "Warp-level Atomic Sequence Ops Forward (CPU)");
}
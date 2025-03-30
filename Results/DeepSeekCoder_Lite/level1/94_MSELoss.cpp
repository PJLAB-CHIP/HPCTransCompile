#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <omp.h>

static const int BLOCK_DIM_X = 16;
static const int BLOCK_DIM_Y = 16;

template <typename scalar_t>
void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;
    const int num_threads_per_block = BLOCK_DIM_X * BLOCK_DIM_Y;

    #pragma omp parallel for reduction(+:thread_sum) schedule(dynamic)
    for (int64_t idx = 0; idx < num_elements; ++idx) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
    }

    // Reduce sum across threads
    double block_sum = 0.0;
    #pragma omp critical
    {
        block_sum = thread_sum;
    }

    // Write result
    sum_out[0] += block_sum;
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Calculate 2D grid dimensions
    int grid_x = std::min(32, (int)ceil(sqrt(num_elements / (BLOCK_DIM_X * BLOCK_DIM_Y))));
    int grid_y = grid_x;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel<scalar_t>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CPU)");
}
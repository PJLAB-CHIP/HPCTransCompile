#include <torch/extension.h>
#include <vector>
#include <limits>
#include <cmath>
#include <omp.h>

template <typename scalar_t>
void unroll_tuned_log_softmax_forward_cpu(
    const scalar_t* input,
    scalar_t* output,
    int dim_size,
    int num_batches) {

    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const scalar_t* input_row = input + batch_idx * dim_size;
        scalar_t* output_row = output + batch_idx * dim_size;

        // Phase 1: Compute the maximum value in the row
        scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
        for (int idx = 0; idx < dim_size; ++idx) {
            scalar_t val = input_row[idx];
            local_max = (val > local_max) ? val : local_max;
        }

        // Phase 2: Compute the sum of exp(x - max_val) for numerical stability
        scalar_t local_sum = 0;
        for (int idx = 0; idx < dim_size; ++idx) {
            // Compute exponentials
            scalar_t exp_val = exp(input_row[idx] - local_max);
            local_sum += exp_val;
        }

        scalar_t log_sum = log(local_sum);

        // Phase 3: Write back the final LogSoftmax values
        for (int idx = 0; idx < dim_size; ++idx) {
            output_row[idx] = (input_row[idx] - local_max) - log_sum;
        }
    }
}

torch::Tensor unroll_tuned_log_softmax_cpu_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(!input.is_cuda(), "input must be a CPU tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
    dim = dim >= 0 ? dim : dim + ndim;

    // Permute input so that the target dimension is the last dimension
    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            permute_dims.push_back(i);
        }
    }
    permute_dims.push_back(dim);

    input = input.permute(permute_dims).contiguous();
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    auto output = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "unroll_tuned_log_softmax_forward_cpu", ([&] {
        unroll_tuned_log_softmax_forward_cpu<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            batch_size);
    }));

    // Inverse permutation to restore original data layout
    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unroll_tuned_log_softmax_cpu_forward, "Unroll Tuned LogSoftmax forward (CPU)");
}

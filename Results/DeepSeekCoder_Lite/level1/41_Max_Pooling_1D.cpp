#include <torch/extension.h>
#include <vector>
#include <omp.h>

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices)
{
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));

    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<std::vector<int>>> thread_max_indices(num_threads);
    std::vector<std::vector<std::vector<float>>> thread_max_values(num_threads);

    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        thread_max_indices[thread_id].resize(batch_size);
        thread_max_values[thread_id].resize(num_channels);
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_channels; ++c) {
                thread_max_indices[thread_id][b].resize(output_length, -1);
                thread_max_values[thread_id][b].resize(output_length, -INFINITY);
            }
        }
    }

    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_channels; ++c) {
            for (int i = 0; i < output_length; ++i) {
                const int input_start = i * stride - padding;
                for (int k = 0; k < kernel_size; ++k) {
                    const int pos = input_start + k * dilation;
                    if (pos >= 0 && pos < input_length) {
                        const int out_idx = b * num_channels * output_length + c * output_length + i;
                        const float val = x[b * num_channels * input_length + c * input_length + pos];
                        if (val > thread_max_values[omp_get_thread_num()][b][i]) {
                            thread_max_values[omp_get_thread_num()][b][i] = val;
                            thread_max_indices[omp_get_thread_num()][b][i] = pos;
                        }
                    }
                }
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_channels; ++c) {
            for (int i = 0; i < output_length; ++i) {
                const int out_idx = b * num_channels * output_length + c * output_length + i;
                output[out_idx] = thread_max_values[omp_get_thread_num()][b][i];
                if (return_indices) {
                    indices[out_idx] = thread_max_indices[omp_get_thread_num()][b][i];
                }
            }
        }
    }

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward (CPU)");
}
#include <torch/extension.h>
#include <omp.h>

void custom_cpu_kernel(
    const float *x,
    const float *weight,
    float *output,
    float scaling_factor,
    int input_size,
    int hidden_size,
    int batch_size) {

    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        float thread_sum = 0.0f;
        int j_per_thread = (hidden_size + omp_get_num_threads() - 1) / omp_get_num_threads();
        int start_j = omp_get_thread_num() * j_per_thread;
        int end_j = std::min((omp_get_thread_num() + 1) * j_per_thread, hidden_size);

        for (int j = start_j; j < end_j; ++j) {
            const float *weight_row = weight + j * input_size;
            float dot = 0.0f;
            for (int k = 0; k < input_size; ++k) {
                dot += x[batch_idx * input_size + k] * weight_row[k];
            }
            thread_sum += dot;
        }

        output[batch_idx] = (thread_sum / 2.0f) * scaling_factor;
    }
}

torch::Tensor forward_cpu(
    torch::Tensor x,
    float scaling_factor,
    torch::Tensor weight) {

    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::zeros({batch_size, 1}, x.options());

    custom_cpu_kernel(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        input_size,
        hidden_size,
        batch_size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Custom forward CPU function");
}
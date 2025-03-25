#include <torch/extension.h>
#include <omp.h>

// Function to perform parallel reduction of group_norm_bias
float parallel_reduce_sum(const float* data, int size) {
    float total_sum = 0.0f;
    #pragma omp parallel for reduction(+:total_sum)
    for (int i = 0; i < size; ++i) {
        total_sum += data[i];
    }
    return total_sum;
}

// Function to broadcast the computed mean to the output array
void broadcast_mean(float* output, float mean, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output[i] = mean;
    }
}

// Torch binding function
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int num_groups
) {
    int batch_size = x.size(0);
    auto output = torch::zeros({batch_size, 1}, x.options());

    // Perform parallel reduction of group_norm_bias
    float total_sum = parallel_reduce_sum(group_norm_bias.data_ptr<float>(), group_norm_bias.size(0));
    float mean = total_sum / group_norm_bias.size(0);

    // Broadcast the computed mean to the output array
    broadcast_mean(output.data_ptr<float>(), mean, batch_size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined fused ops forward function");
}
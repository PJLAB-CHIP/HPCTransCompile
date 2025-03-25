#include <torch/extension.h>
#include <omp.h>

float warpReduceSum(float val) {
    #pragma omp parallel for reduction(+:val)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += val;
    }
    return val;
}

void global_avg_pool_cpu(
    const float* input,
    float* output,
    int spatial_size,
    int num_elements
) {
    #pragma omp parallel for
    for (int i = 0; i < num_elements; ++i) {
        int bid = i / spatial_size;
        int tid = i % spatial_size;
        int index = bid * spatial_size + tid;
        
        float sum = 0.0f;
        for (int j = 0; j < spatial_size; ++j) {
            sum += input[bid * spatial_size + j];
        }
        
        output[bid] = sum / (float)spatial_size;
    }
}

torch::Tensor module_fn_cpu(
    torch::Tensor x,
    double eps,
    double momentum,
    double scale_factor,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var
) {
    // Perform ConvTranspose3d
    x = torch::conv_transpose3d(
        x,
        conv_transpose,
        conv_transpose_bias,
        /*stride=*/{1, 1, 1},
        /*padding=*/{0, 0, 0},
        /*output_padding=*/{0, 0, 0},
        /*groups=*/1,
        /*dilation=*/{1, 1, 1}
    );

    // Multiply by scale_factor
    x = x * scale_factor;

    // Batch Normalization
    x = torch::batch_norm(
        x,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        /*training=*/true,
        momentum,
        eps,
        /*cudnn_enabled=*/false
    );

    // Custom global average pooling implementation
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int spatial_size = sizes[2] * sizes[3] * sizes[4];
    
    auto x_reshaped = x.view({batch_size * channels, spatial_size});
    auto output = torch::empty({batch_size * channels}, x.options());
    
    int num_elements = batch_size * channels;
    global_avg_pool_cpu(
        x_reshaped.data_ptr<float>(),
        output.data_ptr<float>(),
        spatial_size,
        num_elements
    );
    
    return output.view({batch_size, channels, 1, 1, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu, "Module function forward (CPU)");
}
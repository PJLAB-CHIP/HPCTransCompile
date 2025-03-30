#include <torch/extension.h>
#include <vector>
#include <omp.h>

void module_fn_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float scaling_factor,
    const int thread_id)
{
    const int warp_size = 32;
    const int lane_id = thread_id % warp_size;
    const int warp_id = thread_id / warp_size;
    
    int row = warp_id;
    int col = warp_id * warp_size + lane_id;
    
    if (row < batch_size && col < out_features) {
        float val = 0.0f;
        
        // Each warp handles a portion of the reduction
        for (int k = lane_id; k < in_features; k += warp_size) {
            val += x[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Warp-level reduction using shuffle operations
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            val += __builtin_shuffle(val, val, offset);
        }
        
        // First thread in warp has final sum
        if (lane_id == 0) {
            val += bias[col];
            float original_val = val;
            val *= scaling_factor;
            val += original_val;
            out[row * out_features + col] = val;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    const float scaling_factor,
    torch::Tensor weight,
    torch::Tensor bias)
{
    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(weight.is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(bias.is_cpu(), "bias must be a CPU tensor");
    
    auto x_ = x.contiguous();
    auto w_ = weight.contiguous();
    auto b_ = bias.contiguous();

    const int batch_size = x_.size(0);
    const int in_features = x_.size(1);
    const int out_features = w_.size(0);

    auto out = torch::empty({batch_size, out_features}, x_.options());

    #pragma omp parallel for
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < out_features; ++col) {
            int thread_id = omp_get_thread_num();
            module_fn_forward_kernel(
                x_.data_ptr<float>(),
                w_.data_ptr<float>(),
                b_.data_ptr<float>(),
                out.data_ptr<float>(),
                batch_size,
                in_features,
                out_features,
                scaling_factor,
                thread_id
            );
        }
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "module_fn forward (CPU)");
}
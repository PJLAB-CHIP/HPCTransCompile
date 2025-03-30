#include <torch/extension.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <omp.h>

static const int NUM_STREAMS = 4;
static bool streams_created = false;

void create_streams() {
    if (!streams_created) {
        // No-op for CPU implementation
        streams_created = true;
    }
}

void destroy_streams() {
    if (streams_created) {
        // No-op for CPU implementation
        streams_created = false;
    }
}

template <typename scalar_t>
void layernorm_streamed_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int chunk_size,
    const int chunk_offset) {

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    const int outer_size = normalized_size * chunk_size;
    const int instance_idx = chunk_offset;
    
    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;
    
    #pragma omp parallel for
    for (int idx = 0; idx < normalized_size; ++idx) {
        accscalar_t local_sum = 0;
        accscalar_t local_sum_sq = 0;
        
        for (int i = 0; i < chunk_size; ++i) {
            int instance_idx = idx + i * normalized_size;
            accscalar_t val = static_cast<accscalar_t>(in_ptr[instance_idx]);
            local_sum += val;
            local_sum_sq += val * val;
        }
        
        accscalar_t mean = local_sum / chunk_size;
        accscalar_t variance = (local_sum_sq / chunk_size) - (mean * mean);
        accscalar_t inv_std = 1.0f / sqrt(variance + eps);
        
        for (int i = 0; i < chunk_size; ++i) {
            int instance_idx = idx + i * normalized_size;
            accscalar_t val = static_cast<accscalar_t>(in_ptr[instance_idx]);
            accscalar_t normalized = (val - mean) * inv_std;
            out_ptr[instance_idx] = static_cast<scalar_t>(
                normalized * weight[instance_idx] + bias[instance_idx]);
        }
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    create_streams();
    
    auto output = torch::empty_like(x);
    
    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;
    const int chunk_size = (outer_size + NUM_STREAMS - 1) / NUM_STREAMS;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        for (int i = 0; i < NUM_STREAMS; i++) {
            int stream_chunk_size = std::min(chunk_size, outer_size - i * chunk_size);
            if (stream_chunk_size <= 0) break;
            
            layernorm_streamed_kernel<scalar_t>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                static_cast<float>(eps),
                output.data_ptr<scalar_t>(),
                normalized_size,
                chunk_size,
                i * chunk_size);
        }
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward (CPU)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
    // Add cleanup function for streams (no-op for CPU)
    m.def("cleanup", &destroy_streams, "Cleanup CUDA streams (no-op for CPU)");
}
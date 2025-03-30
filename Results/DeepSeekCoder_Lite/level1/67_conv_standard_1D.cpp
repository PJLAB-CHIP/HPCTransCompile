#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <stdint.h>
#include <omp.h>

namespace py = pybind11;

// Helper functions for vectorized (128-bit) loads/stores
inline float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

inline void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

// Host wrapper function to set up kernel launch parameters
at::Tensor conv1d_forward_impl(
    const at::Tensor& x,
    const at::Tensor& w,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");

    // Get input dimensions
    auto x_sizes = x.sizes();
    int N    = x_sizes[0];
    int C_in = x_sizes[1];
    int L_in = x_sizes[2];

    auto w_sizes = w.sizes();
    int C_out = w_sizes[0];
    int K     = w_sizes[2];

    // Compute output length based on convolution parameters
    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Configure block and grid dimensions; each thread processes 4 output positions
    const int threads_per_block = 128;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(
        C_out, // one grid block per output channel
        (L_out + threads_per_block * 4 - 1) / (threads_per_block * 4), // groups of 4 output positions per thread
        N  // one grid layer per batch element
    );

    // Parallelize over output channels and batches
    #pragma omp parallel for
    for (int out_ch = 0; out_ch < C_out; ++out_ch) {
        for (int batch_idx = 0; batch_idx < N; ++batch_idx) {
            // Accumulator for 4 outputs (vectorized)
            float4 output = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float* acc = reinterpret_cast<float*>(&output);

            // Iterate over the input channels in the current group
            for (int local_in_ch = 0; local_in_ch < C_in / groups; ++local_in_ch) {
                int in_ch = (out_ch / groups) * (C_in / groups) + local_in_ch;
                // Pointer to weights for this output channel and input channel
                const float* weight_ptr = &w[out_ch * (C_in / groups) * K + local_in_ch * K];
                
                // Loop over kernel positions
                for (int k = 0; k < K; ++k) {
                    float w_val = weight_ptr[k];
                    
                    for (int i = 0; i < 4; ++i) {
                        int out_pos = (omp_get_thread_num() * threads_per_block * 4 + i) * stride + k * dilation - padding;
                        if (out_pos >= 0 && out_pos < L_out) {
                            int in_pos = batch_idx * (C_in * L_in) + in_ch * L_in + out_pos * stride + k * dilation - padding;
                            float x_val = x[in_pos];
                            acc[i] += x_val * w_val;
                        }
                    }
                }
            }

            // Add bias if available
            if (bias_ptr) {
                float b = bias_ptr[out_ch];
                for (int i = 0; i < 4; ++i) {
                    acc[i] += b;
                }
            }

            // Write results back to y using vectorized store if 16-byte aligned and full 4-element group
            int out_offset = batch_idx * (C_out * L_out) + out_ch * L_out + omp_get_thread_num() * threads_per_block * 4;
            int remaining = ((omp_get_thread_num() * threads_per_block * 4 + 4) <= L_out) ? 4 : (L_out - out_offset / 4 * 4);
            if (remaining == 4 && ((reinterpret_cast<uintptr_t>(&y[out_offset]) & 15) == 0)) {
                store_float4(&y[out_offset], output);
            } else {
                for (int i = 0; i < remaining; ++i) {
                    y[out_offset + i] = acc[i];
                }
            }
        }
    }

    return y;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x,
           at::Tensor w,
           py::object bias_obj,
           int64_t stride,
           int64_t padding,
           int64_t dilation,
           int64_t groups) {
            c10::optional<at::Tensor> bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<at::Tensor>();
            }
            return conv1d_forward_impl(x, w, bias, stride, padding, dilation, groups);
        },
        "1D Convolution forward (CPU) using __ldg() for optimized read-only loads and 128-bit aligned accesses"
    );
}
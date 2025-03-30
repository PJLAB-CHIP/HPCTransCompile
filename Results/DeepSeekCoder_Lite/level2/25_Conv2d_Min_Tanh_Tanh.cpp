#include <torch/extension.h>
#include <cmath>
#include <cstdio>
#include <omp.h>

// This function performs the exact same operation as the CUDA kernel.
void conv_min_tanh_forward_cpu(
    const float* x,    // Input tensor: [B, C_in, H, W]
    const float* weight, // Weight: [C_out, C_in, K, K]
    const float* bias,   // Bias: [C_out]
    float* output,       // Output tensor: [B, 1, H_out, W_out]
    const int batch,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_size,
    const int out_height,
    const int out_width) {

    int num_pixels = batch * out_height * out_width;

    #pragma omp parallel for
    for (int tid = 0; tid < num_pixels; tid++) {
        // Map tid to (b, out_y, out_x)
        int b = tid / (out_height * out_width);
        int rem = tid % (out_height * out_width);
        int out_y = rem / out_width;
        int out_x = rem % out_width;

        float min_val = 1e20f;  // Initialize with a large number
        int kernel_area = kernel_size * kernel_size;

        // Iterate over all output channels uniformly
        for (int oc = 0; oc < out_channels; oc++) {
            // Use __ldg for read-only memory accesses
            float conv_sum = bias[oc];
            
            // Process all input channels
            for (int ic = 0; ic < in_channels; ic++) {
                // Precompute base indices for input and weight
                int base_x = b * (in_channels * in_height * in_width) + ic * (in_height * in_width);
                int base_w = oc * (in_channels * kernel_area) + ic * kernel_area;
                
                // Optimized inner loop: unroll for common kernel size 3 to reduce loop overhead
                if (kernel_size == 3) {
                    int in_y = out_y;
                    int in_x = out_x;
                    conv_sum += x[base_x + (in_y + 0) * in_width + (in_x + 0)] * weight[base_w + 0];
                    conv_sum += x[base_x + (in_y + 0) * in_width + (in_x + 1)] * weight[base_w + 1];
                    conv_sum += x[base_x + (in_y + 0) * in_width + (in_x + 2)] * weight[base_w + 2];
                    conv_sum += x[base_x + (in_y + 1) * in_width + (in_x + 0)] * weight[base_w + 3];
                    conv_sum += x[base_x + (in_y + 1) * in_width + (in_x + 1)] * weight[base_w + 4];
                    conv_sum += x[base_x + (in_y + 1) * in_width + (in_x + 2)] * weight[base_w + 5];
                    conv_sum += x[base_x + (in_y + 2) * in_width + (in_x + 0)] * weight[base_w + 6];
                    conv_sum += x[base_x + (in_y + 2) * in_width + (in_x + 1)] * weight[base_w + 7];
                    conv_sum += x[base_x + (in_y + 2) * in_width + (in_x + 2)] * weight[base_w + 8];
                } else {
                    // Generic convolution for other kernel sizes
                    for (int k = 0; k < kernel_area; k++) {
                        int ky = k / kernel_size;
                        int kx = k - ky * kernel_size;
                        int in_y = out_y + ky;
                        int in_x = out_x + kx;
                        int x_index = base_x + in_y * in_width + in_x;
                        int w_index = base_w + k;
                        conv_sum += x[x_index] * weight[w_index];
                    }
                }
            }
            // Use fminf to avoid branch divergence in the min computation
            min_val = std::fminf(min_val, conv_sum);
        }

        // Apply double tanh activation
        float activated = tanhf(tanhf(min_val));
        
        // Write the result to output. tid already corresponds to the correct output pixel.
        output[tid] = activated;
    }
}

// C++ interface (called from Python via pybind11)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_min_tanh_forward_cpu, "Convolution, min (over channels), and double tanh activation (CPU) with uniform control flow");
}
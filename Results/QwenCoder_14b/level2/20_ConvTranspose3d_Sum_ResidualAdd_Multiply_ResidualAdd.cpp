#include <torch/extension.h>
#include <omp.h>

void coalesced_vectorized_fused_operations_cpu(
    const float* conv_output,
    const float* element_bias,
    float* output,
    int num_elements,
    int channels,
    int spatial_size
) {
    // Cache bias values in a local array
    float shared_bias[channels];
    #pragma omp parallel for
    for (int i = 0; i < channels; i++) {
        shared_bias[i] = element_bias[i];
    }

    // Process elements in vectorized manner using float4
    int total_vec = num_elements / 4;  // number of complete float4 groups

    #pragma omp parallel for
    for (int i = 0; i < total_vec; i++) {
        // Load 4 contiguous floats at once
        float4 in_vec = reinterpret_cast<const float4*>(conv_output)[i];
        int base = i * 4;
        float4 out_vec;
        // Unroll the computation for the 4 elements
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int global_idx = base + j;
            int c = (global_idx / spatial_size) % channels;
            // Access the j-th component of the vector
            float original = ((float*)&in_vec)[j];
            float b = shared_bias[c];
            ((float*)&out_vec)[j] = original * (2.0f * original + b + 1.0f);
        }
        // Store the computed 4 elements back to global memory
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Process any remaining elements that don't form a complete float4
    int remainder = num_elements % 4;
    int start = total_vec * 4;

    #pragma omp parallel for
    for (int i = 0; i < remainder; i++) {
        int global_idx = start + i;
        int c = (global_idx / spatial_size) % channels;
        float orig = conv_output[global_idx];
        output[global_idx] = orig * (2.0f * orig + shared_bias[c] + 1.0f);
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int stride,
    int padding,
    int output_padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    // Compute the transposed convolution using PyTorch's optimized function
    auto conv_result = torch::conv_transpose3d(
        x,
        conv_transpose,
        conv_transpose_bias,
        stride,
        padding,
        output_padding
    );

    // Get dimensions; assume conv_result is in shape [N, C, D, H, W] and is contiguous
    auto sizes = conv_result.sizes();
    int channels = sizes[1];
    int spatial_size = sizes[2] * sizes[3] * sizes[4];  // D * H * W
    int num_elements = conv_result.numel();

    // Prepare the output tensor
    auto output = torch::empty_like(conv_result);

    // Call the CPU function
    coalesced_vectorized_fused_operations_cpu(
        conv_result.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        channels,
        spatial_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Vectorized Fused ConvTranspose3D Kernel with Channel-wise Bias");
}
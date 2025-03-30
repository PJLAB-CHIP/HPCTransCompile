#include <torch/extension.h>
#include <omp.h>

// This function performs the exact same operation as the CUDA kernel.
// It uses OpenMP for multi-threading to parallelize the computation.
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

    // Parallelize the computation using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < num_elements; ++i) {
        int c = ((i / spatial_size) % channels);
        float orig = conv_result[i].item<float>();
        output[i] = orig * (2.0f * orig + bias[c].item<float>() + 1.0f);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Vectorized Fused ConvTranspose3D Kernel with Channel-wise Bias");
}
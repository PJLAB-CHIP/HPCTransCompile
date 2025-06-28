#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

// Block size for spatial dimension processing
#define BLOCK_SIZE 256

// Constant memory for HardSwish parameters
// d_hswish_constants[0] = offset (3.0f), d_hswish_constants[1] = cap (6.0f)
__constant__ float d_hswish_constants[2];
__constant__ float d_hswish_div;  // = 1/6.0f

// Initialize constant memory values (to be called once)
void initialize_constants() {
    float h_constants[2] = {3.0f, 6.0f};
    cudaMemcpyToSymbol(d_hswish_constants, h_constants, 2 * sizeof(float));
    float div = 1.0f / 6.0f;
    cudaMemcpyToSymbol(d_hswish_div, &div, sizeof(float));
}

// Fused kernel: applies HardSwish, ReLU, and Softmax in three passes over the channel dimension.
// It uses __ldg() for read-only global memory loads to leverage texture cache and assumes that the
// input data is allocated with 128-bit alignment. Each thread processes one spatial index in one batch.

__global__ void ldg_aligned_fused_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int batch_size,
                                           int channels,
                                           int spatial_size) {
    // Calculate spatial index and batch index
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    if (spatial_idx >= spatial_size || batch_idx >= batch_size) return;

    float max_val = -FLT_MAX;

    // Pass 1: Compute maximum activation value across channels for numerical stability
    // Activation: act = fmax( x * min(max(x+3,0),6) / 6, 0 )
    for (int c = 0; c < channels; ++c) {
        int idx = (batch_idx * channels + c) * spatial_size + spatial_idx;
        // Use __ldg() for read-only access; assumes input is 128-bit aligned when possible
        float x = __ldg(&input[idx]);
        float relu6 = fminf(fmaxf(x + d_hswish_constants[0], 0.0f), d_hswish_constants[1]);
        float hswish = x * relu6 * d_hswish_div;
        float act = fmaxf(hswish, 0.0f);
        if (act > max_val) {
            max_val = act;
        }
    }

    float sum_exp = 0.0f;

    // Pass 2: Compute exponentials and accumulate the sum, store exp values temporarily in output
    for (int c = 0; c < channels; ++c) {
        int idx = (batch_idx * channels + c) * spatial_size + spatial_idx;
        float x = __ldg(&input[idx]);
        float relu6 = fminf(fmaxf(x + d_hswish_constants[0], 0.0f), d_hswish_constants[1]);
        float hswish = x * relu6 * d_hswish_div;
        float act = fmaxf(hswish, 0.0f);
        float exp_val = expf(act - max_val);
        sum_exp += exp_val;
        output[idx] = exp_val;
    }

    // Pass 3: Normalize the exponentials to obtain softmax probabilities
    for (int c = 0; c < channels; ++c) {
        int idx = (batch_idx * channels + c) * spatial_size + spatial_idx;
        output[idx] = output[idx] / sum_exp;
    }
}

// Module forward function: combines conv3d, the fused activation and softmax kernel, and mean reduction
// The softmax is applied over the channel dimension after reformatting the tensor.

torch::Tensor module_forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias) {
    // Initialize constant memory once
    static bool constants_initialized = false;
    if (!constants_initialized) {
        initialize_constants();
        constants_initialized = true;
    }

    // Ensure tensors are contiguous and on CUDA
    x = x.contiguous().cuda();
    conv_weight = conv_weight.contiguous().cuda();
    conv_bias = conv_bias.contiguous().cuda();

    // Perform 3D convolution via PyTorch's conv3d
    x = torch::conv3d(x, conv_weight, conv_bias);

    // Retrieve tensor dimensions
    int64_t batch_size = x.size(0);
    int64_t channels = x.size(1);
    int64_t depth = x.size(2);
    int64_t height = x.size(3);
    int64_t width = x.size(4);
    int64_t spatial_size = depth * height * width;

    // Allocate intermediate tensor for softmax result
    torch::Tensor x_softmax = torch::empty_like(x);

    // Launch kernel: 2D grid (spatial index and batch index)
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((spatial_size + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);

    ldg_aligned_fused_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                                  x_softmax.data_ptr<float>(),
                                                  batch_size,
                                                  channels,
                                                  spatial_size);

    // Reshape back to original dimensions and compute mean over spatial dims
    torch::Tensor output = x_softmax.view({batch_size, channels, depth, height, width}).mean({2, 3, 4});
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_forward, "Fused CUDA module forward with __ldg() and aligned global memory accesses");
}

#include <torch/extension.h>
#include <vector>
#include <omp.h>

void elementwise_add_cpu(float* out, const float* sum_weight, int64_t num_elements) {
    #pragma omp parallel for
    for(int64_t i = 0; i < num_elements; ++i) {
        out[i] += *sum_weight;
    }
}

void layer_norm_forward_cpu(float* X, float* Y, const float* gamma, const float* beta, int num_features, int feature_stride, float epsilon) {
    #pragma omp parallel for
    for(int64_t bid = 0; bid < num_features; ++bid) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for(int64_t tid = 0; tid < feature_stride; ++tid) {
            float val = X[bid * feature_stride + tid];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / feature_stride;
        float var = sum_sq / feature_stride - mean * mean;
        float inv_std = 1.0f / sqrtf(var + epsilon);
        
        for(int64_t tid = 0; tid < feature_stride; ++tid) {
            float val = (X[bid * feature_stride + tid] - mean) * inv_std;
            Y[bid * feature_stride + tid] = val * gamma[tid] + beta[tid];
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_transpose_weight,
    torch::Tensor conv_transpose_bias,
    torch::Tensor sum_weight,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> pool_kernel_size,
    std::vector<int64_t> norm_shape
) {
    at::IntArrayRef strideRef(stride);
    at::IntArrayRef paddingRef(padding);
    at::IntArrayRef outputPaddingRef(output_padding);
    at::IntArrayRef poolKernelRef(pool_kernel_size);
    at::IntArrayRef normShapeRef(norm_shape);

    // 1. 3D transposed convolution
    auto out = at::conv_transpose3d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        strideRef,
        paddingRef,
        outputPaddingRef,
        /*groups=*/1,
        /*dilation=*/1
    );

    // 2. Optimized elementwise addition
    int64_t num_elements = out.numel();
    elementwise_add_cpu(out.data_ptr<float>(), sum_weight.data_ptr<float>(), num_elements);

    // 3. Custom layer normalization
    const int num_features = norm_shape.back();
    const int feature_stride = num_features;
    layer_norm_forward_cpu(out.data_ptr<float>(), out.data_ptr<float>(), norm_weight.data_ptr<float>(), norm_bias.data_ptr<float>(), num_features, feature_stride, 1e-5);

    // 4. 3D average pooling
    out = at::avg_pool3d(
        out,
        poolKernelRef,
        poolKernelRef,
        /*padding=*/{0, 0, 0},
        /*ceil_mode=*/false,
        /*count_include_pad=*/true
    );

    // 5. GELU activation
    out = at::gelu(out);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "module_fn forward (CPU)");
}
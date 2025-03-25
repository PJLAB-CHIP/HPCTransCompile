#include <torch/extension.h>
#include <vector>
#include <cfloat>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

template<int KERNEL_SIZE>
float compute_conv_aligned(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int input_offset,
    const int weight_offset,
    const int in_w,
    const int oh,
    const int ow) 
{
    float sum = 0.0f;
    
    for(int i = 0; i < KERNEL_SIZE; ++i) {
        const int in_row_offset = input_offset + (oh + i) * in_w;
        const int weight_row_offset = weight_offset + i * KERNEL_SIZE;
        
        for(int j = 0; j < KERNEL_SIZE; ++j) {
            sum += input[in_row_offset + ow + j] * weight[weight_row_offset + j];
        }
    }
    return sum;
}

void conv_scale_min_cpu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float scale_factor,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w) 
{
    #pragma omp parallel for collapse(3)
    for(int n = 0; n < batch; ++n) {
        for(int oh = 0; oh < out_h; ++oh) {
            for(int ow = 0; ow < out_w; ++ow) {
                float min_val = FLT_MAX;
                const int in_area = in_h * in_w;
                const int input_batch_offset = n * in_channels * in_area;
                
                for(int oc = 0; oc < out_channels; ++oc) {
                    float conv_sum = bias[oc];
                    const int weight_oc_offset = oc * in_channels * KERNEL_SIZE * KERNEL_SIZE;
                    
                    for(int ic = 0; ic < in_channels; ++ic) {
                        const int input_ic_offset = input_batch_offset + ic * in_area;
                        const int weight_ic_offset = weight_oc_offset + ic * KERNEL_SIZE * KERNEL_SIZE;
                        
                        conv_sum += compute_conv_aligned<KERNEL_SIZE>(
                            input,
                            weight,
                            input_ic_offset,
                            weight_ic_offset,
                            in_w,
                            oh,
                            ow
                        );
                    }
                    
                    conv_sum *= scale_factor;
                    min_val = fminf(min_val, conv_sum);
                }
                
                const int out_idx = n * out_h * out_w + oh * out_w + ow;
                output[out_idx] = min_val;
            }
        }
    }
}

at::Tensor forward(at::Tensor x, at::Tensor conv_weight, at::Tensor conv_bias, float scale_factor) {
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(conv_weight);
    CHECK_CONTIGUOUS(conv_bias);

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int out_channels = conv_weight.size(0);
    const int kernel_size = conv_weight.size(2);
    
    const int out_h = in_h - kernel_size + 1;
    const int out_w = in_w - kernel_size + 1;

    auto options = x.options();
    at::Tensor output = at::zeros({batch, 1, out_h, out_w}, options);

    conv_scale_min_cpu(
        x.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        scale_factor,
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-aligned convolution with scaling and min reduction (CPU)");
}
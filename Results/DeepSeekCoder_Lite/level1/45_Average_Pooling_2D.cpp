#include <torch/extension.h>
#include <omp.h>

// Forward function exposed to PyTorch

torch::Tensor manual_unroll_avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto x_cont = x.contiguous();
    auto options = x.options();
    auto output = torch::empty({N, C, outH, outW}, options);
    
    const auto* input = x_cont.data_ptr<float>();
    auto* output_data = output.data_ptr<float>();
    
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int out_y = 0; out_y < outH; ++out_y) {
                for (int out_x = 0; out_x < outW; ++out_x) {
                    int in_x_start = out_x * stride - padding;
                    int in_y_start = out_y * stride - padding;
                    float sum = 0.0f;
                    bool fully_inside = (in_x_start >= 0) && (in_y_start >= 0) &&
                                        ((in_x_start + kernel_size) <= W) &&
                                        ((in_y_start + kernel_size) <= H);
                    int out_index = ((n * C + c) * outH + out_y) * outW + out_x;
                    
                    if (kernel_size == 3 && fully_inside) {
                        int base = (n * C + c) * H;
                        int ix = in_x_start;
                        int row0 = base + in_y_start;
                        int row1 = base + in_y_start + 1;
                        int row2 = base + in_y_start + 2;
                        sum = input[row0 * W + ix]     + input[row0 * W + ix + 1]     + input[row0 * W + ix + 2] +
                              input[row1 * W + ix]     + input[row1 * W + ix + 1]     + input[row1 * W + ix + 2] +
                              input[row2 * W + ix]     + input[row2 * W + ix + 1]     + input[row2 * W + ix + 2];
                    } else {
                        for (int ky = 0; ky < kernel_size; ky++) {
                            int y = in_y_start + ky;
                            for (int kx = 0; kx < kernel_size; kx++) {
                                int x = in_x_start + kx;
                                if (y >= 0 && y < H && x >= 0 && x < W) {
                                    int index_in = ((n * C + c) * H + y) * W + x;
                                    sum += input[index_in];
                                }
                            }
                        }
                    }
                    
                    output_data[out_index] = sum / static_cast<float>(kernel_size * kernel_size);
                }
            }
        }
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &manual_unroll_avg_pool2d_forward, "Manual Unroll 2D Average Pooling forward (CPU)");
}
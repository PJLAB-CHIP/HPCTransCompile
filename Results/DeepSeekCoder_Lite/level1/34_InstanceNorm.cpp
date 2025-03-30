#include <torch/extension.h>
#include <vector>
#include <omp.h>

__inline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__inline__ float blockReduceSum(float val) {
    static __thread float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    if (weight.defined()) TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");

    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    auto y = torch::empty_like(x);

    #pragma omp parallel for
    for (int instance_id = 0; instance_id < N * C; ++instance_id) {
        int n = instance_id / C;
        int c = instance_id % C;
        int HW = H * W;

        const float* x_ptr = x.data_ptr<float>() + (n * C + c) * HW;
        float* y_ptr = y.data_ptr<float>() + (n * C + c) * HW;

        // Load data into shared memory and compute partial sums
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int i = threadIdx.x; i < HW; i += blockDim.x) {
            float val = x_ptr[i];
            sum += val;
            sum_sq += val * val;
        }

        // Reduce sums across thread
        sum = blockReduceSum(sum);
        sum_sq = blockReduceSum(sum_sq);

        __shared__ float mean_sh;
        __shared__ float invstd_sh;

        if (threadIdx.x == 0) {
            mean_sh = sum / HW;
            float var = (sum_sq / HW) - (mean_sh * mean_sh);
            var = (var < 0.f) ? 0.f : var;
            invstd_sh = 1.0f / sqrt(var + eps);
        }
        __syncthreads();

        // Load scale and bias once per thread if they exist
        float scale = weight ? weight.data_ptr<float>()[c] : 1.0f;
        float shift = bias ? bias.data_ptr<float>()[c] : 0.0f;

        // Normalize using shared memory data
        for (int i = threadIdx.x; i < HW; i += blockDim.x) {
            float val = x_ptr[i];
            y_ptr[i] = (val - mean_sh) * invstd_sh * scale + shift;
        }
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CPU)");
}
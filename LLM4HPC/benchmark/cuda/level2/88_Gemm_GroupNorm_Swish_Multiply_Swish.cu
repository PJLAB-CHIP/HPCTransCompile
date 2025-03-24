#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// CUDA forward declarations
torch::Tensor module_fn_cuda_forward(
    torch::Tensor x,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    torch::Tensor multiply_weight,
    int64_t num_groups
);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Helper function for warp reduction
template <typename scalar_t>
__inline__ __device__
scalar_t warpReduceSum(scalar_t val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename scalar_t>
__inline__ __device__
scalar_t blockReduceSum(scalar_t val) {
    __shared__ scalar_t shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x/warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

template<typename scalar_t>
__global__ void module_fn_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ group_norm_weight,
    const scalar_t* __restrict__ group_norm_bias,
    const scalar_t* __restrict__ multiply_weight,
    const int C,
    const int channels_per_group,
    const int chunk_size
) {
    const int chunk_idx = blockIdx.x / chunk_size;
    const int local_n = blockIdx.x % chunk_size;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int n = chunk_idx * chunk_size + local_n;

    __shared__ scalar_t mean_shared;
    __shared__ scalar_t var_shared;

    scalar_t sum = 0.0f;
    scalar_t sumsq = 0.0f;

    #pragma unroll 4
    for (int c = tid; c < channels_per_group; c += blockDim.x) {
        const int channel_idx = g * channels_per_group + c;
        const int idx = n * C + channel_idx;
        scalar_t val = __ldg(&x[idx]);
        sum += val;
        sumsq += val * val;
    }

    sum = blockReduceSum(sum);
    sumsq = blockReduceSum(sumsq);

    if (threadIdx.x == 0) {
        mean_shared = sum / channels_per_group;
        var_shared = sumsq / channels_per_group - mean_shared * mean_shared + 1e-5f;
    }
    __syncthreads();

    const scalar_t mean = mean_shared;
    const scalar_t inv_std = rsqrtf(var_shared);

    #pragma unroll 4
    for (int c = tid; c < channels_per_group; c += blockDim.x) {
        const int channel_idx = g * channels_per_group + c;
        const int idx = n * C + channel_idx;
        
        scalar_t val = x[idx];
        scalar_t gamma = group_norm_weight[channel_idx];
        scalar_t beta = group_norm_bias[channel_idx];
        scalar_t w = multiply_weight[channel_idx];

        scalar_t y = (val - mean) * inv_std;
        y = gamma * y + beta;

        scalar_t sigmoid_y = 1.0f / (1.0f + expf(-y));
        y = y * sigmoid_y;

        y = y * w;

        sigmoid_y = 1.0f / (1.0f + expf(-y));
        y = y * sigmoid_y;

        output[idx] = y;
    }
}

torch::Tensor module_fn_cuda_forward(
    torch::Tensor x,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    torch::Tensor multiply_weight,
    int64_t num_groups
) {
    CHECK_INPUT(x);
    CHECK_INPUT(gemm_weight);
    CHECK_INPUT(gemm_bias);
    CHECK_INPUT(group_norm_weight);
    CHECK_INPUT(group_norm_bias);
    CHECK_INPUT(multiply_weight);

    const int NUM_STREAMS = 4;
    std::vector<at::cuda::CUDAStream> streams;
    for(int i = 0; i < NUM_STREAMS; i++) {
        streams.push_back(at::cuda::getStreamFromPool());
    }

    auto x_linear = torch::addmm(gemm_bias, x, gemm_weight.t());
    auto output = torch::empty_like(x_linear);

    auto N = x_linear.size(0);
    auto C = x_linear.size(1);
    int channels_per_group = C / num_groups;
    
    const int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    
    for(int i = 0; i < NUM_STREAMS; i++) {
        int start_n = i * chunk_size;
        int end_n = std::min((i + 1) * chunk_size, (int)N);
        if(start_n >= end_n) continue;

        auto stream = streams[i];
        // at::cuda::CUDAStreamGuard guard(stream);  // Removed stream guard, as we directly launch kernels on provided streams

        dim3 blocks(end_n - start_n, num_groups);
        int threads = std::min(channels_per_group, 1024);

        AT_DISPATCH_FLOATING_TYPES(x_linear.scalar_type(), "module_fn_cuda_forward", ([&] {
            module_fn_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x_linear.data_ptr<scalar_t>() + start_n * C,
                output.data_ptr<scalar_t>() + start_n * C,
                group_norm_weight.data_ptr<scalar_t>(),
                group_norm_bias.data_ptr<scalar_t>(),
                multiply_weight.data_ptr<scalar_t>(),
                C,
                channels_per_group,
                chunk_size
            );
        }));
    }

    // Synchronize all streams
    for(auto& stream : streams) {
        stream.synchronize();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cuda_forward, "Module function forward");
}
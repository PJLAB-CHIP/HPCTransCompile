#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void block_tuned_kernel(
    const scalar_t* __restrict__ x_linear,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bn_weight,
    const scalar_t* __restrict__ bn_bias,
    scalar_t* __restrict__ bn_running_mean,
    scalar_t* __restrict__ bn_running_var,
    const scalar_t* __restrict__ add_bias,
    const float bn_eps,
    const float bn_momentum,
    const float divide_value,
    const int batch_size,
    const int out_features) {

    extern __shared__ float shared_data[];
    float* s_sum = shared_data;
    float* s_sumsq = &shared_data[128];

    const int f = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (f >= out_features) return;

    float4 local_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 local_sumsq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const int vec_size = 4;
    const int vec_elements = (batch_size / vec_size) * vec_size;
    
    #pragma unroll 4
    for (int i = tid * vec_size; i < vec_elements; i += blockDim.x * vec_size) {
        float4 values;
        values.x = static_cast<float>(x_linear[i * out_features + f]);
        values.y = static_cast<float>(x_linear[(i + 1) * out_features + f]);
        values.z = static_cast<float>(x_linear[(i + 2) * out_features + f]);
        values.w = static_cast<float>(x_linear[(i + 3) * out_features + f]);
        
        local_sum.x += values.x;
        local_sum.y += values.y;
        local_sum.z += values.z;
        local_sum.w += values.w;
        
        local_sumsq.x += values.x * values.x;
        local_sumsq.y += values.y * values.y;
        local_sumsq.z += values.z * values.z;
        local_sumsq.w += values.w * values.w;
    }

    for (int i = vec_elements + tid; i < batch_size; i += blockDim.x) {
        float val = static_cast<float>(x_linear[i * out_features + f]);
        local_sum.x += val;
        local_sumsq.x += val * val;
    }

    float thread_sum = local_sum.x + local_sum.y + local_sum.z + local_sum.w;
    float thread_sumsq = local_sumsq.x + local_sumsq.y + local_sumsq.z + local_sumsq.w;

    s_sum[tid] = thread_sum;
    s_sumsq[tid] = thread_sumsq;
    __syncthreads();

    #pragma unroll
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vs_sum = s_sum;
        volatile float* vs_sumsq = s_sumsq;
        if (blockDim.x >= 64) { vs_sum[tid] += vs_sum[tid + 32]; vs_sumsq[tid] += vs_sumsq[tid + 32]; }
        if (blockDim.x >= 32) { vs_sum[tid] += vs_sum[tid + 16]; vs_sumsq[tid] += vs_sumsq[tid + 16]; }
        if (blockDim.x >= 16) { vs_sum[tid] += vs_sum[tid + 8];  vs_sumsq[tid] += vs_sumsq[tid + 8]; }
        if (blockDim.x >= 8)  { vs_sum[tid] += vs_sum[tid + 4];  vs_sumsq[tid] += vs_sumsq[tid + 4]; }
        if (blockDim.x >= 4)  { vs_sum[tid] += vs_sum[tid + 2];  vs_sumsq[tid] += vs_sumsq[tid + 2]; }
        if (blockDim.x >= 2)  { vs_sum[tid] += vs_sum[tid + 1];  vs_sumsq[tid] += vs_sumsq[tid + 1]; }
    }

    if (tid == 0) {
        float mean = s_sum[0] / batch_size;
        float var = (s_sumsq[0] / batch_size) - (mean * mean);
        
        bn_running_mean[f] = bn_running_mean[f] * (1 - bn_momentum) + mean * bn_momentum;
        bn_running_var[f] = bn_running_var[f] * (1 - bn_momentum) + var * bn_momentum;
        
        s_sum[0] = mean;
        s_sumsq[0] = var;
    }
    __syncthreads();

    const float mean = s_sum[0];
    const float var = s_sumsq[0];
    const float inv_std = rsqrtf(var + bn_eps);
    const float gamma = bn_weight[f];
    const float beta = bn_bias[f];
    const float extra_bias = add_bias[0];

    #pragma unroll 4
    for (int i = tid * vec_size; i < vec_elements; i += blockDim.x * vec_size) {
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
            const int idx = (i + j) * out_features + f;
            float val = static_cast<float>(x_linear[idx]);
            float normalized = (val - mean) * inv_std;
            float transformed = fmaf(normalized, gamma, beta) + extra_bias;
            float divided = transformed / divide_value;
            output[idx] = static_cast<scalar_t>(divided / (1.0f + expf(-divided)));
        }
    }

    for (int i = vec_elements + tid; i < batch_size; i += blockDim.x) {
        const int idx = i * out_features + f;
        float val = static_cast<float>(x_linear[idx]);
        float normalized = (val - mean) * inv_std;
        float transformed = fmaf(normalized, gamma, beta) + extra_bias;
        float divided = transformed / divide_value;
        output[idx] = static_cast<scalar_t>(divided / (1.0f + expf(-divided)));
    }
}

torch::Tensor module_fn_cuda(
    torch::Tensor x,
    float bn_eps,
    float bn_momentum,
    float divide_value,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor add_bias) {

    const auto batch_size = x.size(0);
    const auto out_features = weight.size(0);

    auto x_linear = torch::addmm(bias, x, weight.t());
    auto output = torch::empty_like(x_linear);

    const int threads = 128;
    const int blocks = out_features;
    const size_t shared_mem_size = 2 * threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x_linear.scalar_type(), "block_tuned_kernel", ([&] {
        block_tuned_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x_linear.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            bn_weight.data_ptr<scalar_t>(),
            bn_bias.data_ptr<scalar_t>(),
            bn_running_mean.data_ptr<scalar_t>(),
            bn_running_var.data_ptr<scalar_t>(),
            add_bias.data_ptr<scalar_t>(),
            bn_eps,
            bn_momentum,
            divide_value,
            batch_size,
            out_features);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cuda, "Block tuned forward (CUDA)");
}
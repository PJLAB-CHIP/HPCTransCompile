#include <torch/extension.h>
#include <omp.h>
#include <cmath>

template<unsigned int blockSize>
__forceinline__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

void kl_div_kernel_stage1(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ block_results,
    const int n,
    const int threads,
    const int blocks) {
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int block_size = 256;
        int num_warps = (block_size + 31) / 32;
        __shared__ float sdata[256];
        
        float thread_sum = 0.0f;
        
        for (int i = tid; i < n; i += threads) {
            float log_pred = log_predictions[i];
            float target = targets[i];
            thread_sum += std::exp(log_pred) - target * log_pred;
        }
        
        sdata[tid] = thread_sum;
        __sync_thread();
        
        if (num_warps >= 8) {
            if (tid < 128) sdata[tid] += sdata[tid + 128];
            __sync_thread();
        }
        if (num_warps >= 4) {
            if (tid < 64) sdata[tid] += sdata[tid + 64];
            __sync_thread();
        }
        if (num_warps >= 2) {
            if (tid < 32) warpReduce<256>(sdata, tid);
            __sync_thread();
        }
        if (tid == 0) block_results[blockIdx.x] = sdata[0];
    }
}

void kl_div_kernel_stage2(
    const float* __restrict__ block_results,
    float* __restrict__ output,
    const int num_blocks,
    const float normalizer) {
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        __shared__ float sdata[256];
        
        float sum = 0.0f;
        for (int i = tid; i < num_blocks; i += 256) {
            sum += block_results[i];
        }
        
        sdata[tid] = sum;
        __sync_thread();
        
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __sync_thread();
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __sync_thread();
        if (tid < 32) warpReduce<256>(sdata, tid);
        __sync_thread();
        
        if (tid == 0) {
            output[0] = sdata[0] * normalizer;
        }
    }
}

torch::Tensor kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = std::min((n + threads * 8 - 1) / (threads * 8), 1024);
    const float normalizer = 1.0f / static_cast<float>(n);
    
    auto block_results = torch::empty({blocks}, log_predictions.options());
    
    kl_div_kernel_stage1(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n,
        threads,
        blocks
    );
    
    kl_div_kernel_stage2(
        block_results.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks,
        normalizer
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_forward, "KL divergence forward (CPU)");
}
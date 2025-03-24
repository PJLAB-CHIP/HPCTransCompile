/*
   Fused Forward CUDA Kernel
   This kernel fuses the series of operations from the original implementations:
   1. GEMM (matrix multiplication with bias) 
   2. Subtraction of a per-column constant
   3. Global average pooling
   4. LogSumExp (which is mathematically the identity in this case)
   5. GELU activation
   6. Residual addition with the original input

   Observation:
   The original sequence computes, for each row i and each column j:

       gemm_out[i,j] = dot(x[i,:], weight[j,:]) + bias[j] - subtract[j]
       pool[i] = (1/out_features) * sum_j gemm_out[i,j]
       pool[i] = gelu(pool[i])
       out[i,k] = original_x[i,k] + pool[i]

   Notice that the sum over j can be re-ordered as:

       pool[i] = (1/out_features) * ( dot(x[i,:], sum_{j} weight[j,:]) + sum_{j}(bias[j]-subtract[j]) )
                = ( dot(x[i,:], weight_sum) + constant ) / out_features

   where:
       weight_sum[k] = sum_{j=0}^{out_features-1} weight[j * in_features + k]
       constant = sum_{j=0}^{out_features-1} (bias[j] - subtract[j])

   This transformation allows us to replace the heavy GEMM over (batch_size x out_features) with
   a fast dot product per row over in_features elements. Then, after applying GELU on the pooled
   scalar and adding back via a residual connection, we obtain the same overall result as the original.

   This implementation precomputes weight_sum and constant (using PyTorch tensor operations which run on GPU),
   and then launches a fused CUDA kernel that, for each row, computes the dot product x[i] * weight_sum, 
   applies the necessary normalization, GELU activation, and broadcasts the result as a residual add to x[i].

   The fused kernel uses one block per row and a shared memory reduction for computing the dot product.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

//------------------------------------------------------------------------------
// GELU approximation function
__device__ float gelu_approx(float val) {
    const float kAlpha = 0.044715f;
    const float kBeta  = 0.7978845608f; // sqrt(2/M_PI)
    float inner = kBeta * (val + kAlpha * val * val * val);
    float cdf   = 0.5f * (1.0f + tanhf(inner));
    return val * cdf;
}

//------------------------------------------------------------------------------
// Fused kernel: Computes the dot product of x[i] and weight_sum with a reduction,
// applies normalization using out_features and constant, then applies GELU,
// and finally performs a residual add with x to produce the final output.
// Each block processes one row.
__global__ void fused_forward_kernel(
    const float* __restrict__ x,            // Input x: shape (batch_size, in_features)
    const float* __restrict__ weight_sum,     // Precomputed weight_sum: shape (in_features)
    float constant,                           // Precomputed constant: sum(bias - subtract)
    float* __restrict__ out,                  // Output: shape (batch_size, in_features)
    int batch_size,
    int in_features,
    int out_features                        // Needed for normalization
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float sdata[]; // Shared memory for reduction
    float sum_val = 0.0f;
    
    // Each thread processes a subset of the in_features dimension
    for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
         float x_val = x[row * in_features + k];
         float ws = weight_sum[k];
         sum_val += x_val * ws;
    }
    sdata[threadIdx.x] = sum_val;
    __syncthreads();

    // Reduction in shared memory to compute the dot product
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
         if (threadIdx.x < stride)
              sdata[threadIdx.x] += sdata[threadIdx.x + stride];
         __syncthreads();
    }
    float pool_val = sdata[0];

    // Thread 0 normalizes the sum, applies GELU, and writes back to shared memory
    if (threadIdx.x == 0) {
         pool_val = (pool_val + constant) / static_cast<float>(out_features);
         pool_val = gelu_approx(pool_val);
         sdata[0] = pool_val; // Broadcast the result
    }
    __syncthreads();
    pool_val = sdata[0];

    // Broadcast residual addition: each thread adds pool_val to the corresponding
    // element of the original input x to produce out.
    for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
         out[row * in_features + k] = x[row * in_features + k] + pool_val;
    }
}

//------------------------------------------------------------------------------
// Forward function for the fused kernel
// Precomputes the necessary reductions (weight_sum and constant) and launches the fused kernel.

torch::Tensor forward_cuda_fused(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& subtract
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(subtract.is_cuda(), "subtract must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (batch_size x in_features)");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D (out_features x in_features)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_features)");
    TORCH_CHECK(subtract.dim() == 1, "subtract must be 1D (out_features)");

    int64_t batch_size  = x.size(0);
    int64_t in_features = x.size(1);
    int64_t out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "weight.shape[1] must match x.shape[1]");
    TORCH_CHECK(bias.size(0) == out_features, "bias.shape[0] must match weight.shape[0]");
    TORCH_CHECK(subtract.size(0) == out_features, "subtract.shape[0] must match weight.shape[0]");

    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto subtract_contig = subtract.contiguous();

    // Precompute weight_sum: sum over rows of weight (weight is out_features x in_features)
    // weight_sum will have shape (in_features,)
    auto weight_sum = torch::sum(weight_contig, 0);

    // Precompute constant = sum(bias - subtract) [a scalar]
    auto constant_tensor = torch::sum(bias_contig - subtract_contig);
    float constant = constant_tensor.item<float>();

    // Allocate output tensor (same shape as x)
    auto out = torch::empty({batch_size, in_features}, x.options());

    int threads = 256;
    int blocks = batch_size; // One block per row in x
    size_t shared_mem_bytes = threads * sizeof(float);
    
    fused_forward_kernel<<<blocks, threads, shared_mem_bytes>>>(
        x_contig.data_ptr<float>(),
        weight_sum.data_ptr<float>(),
        constant,
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return out;
}

//------------------------------------------------------------------------------
// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda_fused, "Fused Forward CUDA Kernel");
}

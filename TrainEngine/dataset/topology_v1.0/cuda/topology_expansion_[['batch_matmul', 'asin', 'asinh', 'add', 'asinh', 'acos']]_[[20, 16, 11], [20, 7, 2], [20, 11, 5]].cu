extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_add, float* __restrict__ ph_0) {
  T_add[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (asinhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) + ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(180) default_function_kernel_3(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN[1];
  __shared__ float ph_3_shared[180];
  T_batch_matmul_NN[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    ph_3_shared[((int)threadIdx.x)] = ph_3[((((((int)threadIdx.x) / 20) * 40) + (k_outer_outer * 20)) + (((int)threadIdx.x) % 20))];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      T_batch_matmul_NN[0] = (T_batch_matmul_NN[0] + (ph_0[((((((((int)threadIdx.x) / 20) * 64) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.x) % 20) / 5) * 8)) + (k_outer_outer * 4)) + k_outer_inner)] * ph_3_shared[((((((int)threadIdx.x) / 20) * 20) + (k_outer_inner * 5)) + (((int)threadIdx.x) % 5))]));
    }
  }
  compute[((((((int)threadIdx.x) / 20) * 40) + (((int)blockIdx.x) * 20)) + (((int)threadIdx.x) % 20))] = acosf(T_batch_matmul_NN[0]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinhf(asinhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}


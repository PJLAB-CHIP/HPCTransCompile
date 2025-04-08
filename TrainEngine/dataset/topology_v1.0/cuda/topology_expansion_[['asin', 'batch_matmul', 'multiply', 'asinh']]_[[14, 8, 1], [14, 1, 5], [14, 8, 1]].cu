extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(15) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN_local[1];
  __shared__ float ph_3_shared[20];
  T_batch_matmul_NN_local[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 10) {
      *(float2*)(ph_3_shared + (((int)threadIdx.x) * 2)) = *(float2*)(ph_3 + ((k_outer_outer * 20) + (((int)threadIdx.x) * 2)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 2; ++k_inner) {
        T_batch_matmul_NN_local[0] = (T_batch_matmul_NN_local[0] + (ph_0[(((((((int)blockIdx.x) * 24) + ((((int)threadIdx.x) / 5) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner)] * ph_3_shared[(((k_outer_inner * 10) + (k_inner * 5)) + (((int)threadIdx.x) % 5))]));
      }
    }
  }
  T_batch_matmul_NN[((((int)blockIdx.x) * 15) + ((int)threadIdx.x))] = T_batch_matmul_NN_local[0];
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ T_multiply, float* __restrict__ ph_0, float* __restrict__ ph_5) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 9) {
    T_multiply[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] * ph_5[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_3(float* __restrict__ compute, float* __restrict__ ph_5) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = asinhf(ph_5[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
  }
}


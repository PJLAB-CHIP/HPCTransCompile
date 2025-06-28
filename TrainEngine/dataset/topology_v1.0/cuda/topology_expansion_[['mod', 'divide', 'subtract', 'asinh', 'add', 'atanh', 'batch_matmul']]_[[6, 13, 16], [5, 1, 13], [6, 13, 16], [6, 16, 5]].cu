extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = atanhf(fmodf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(50) default_function_kernel_3(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_10, float* __restrict__ ph_3) {
  float T_batch_matmul_NN_local[1];
  __shared__ float ph_10_shared[10];
  T_batch_matmul_NN_local[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 10) {
      ph_10_shared[((int)threadIdx.x)] = ph_10[((((((int)blockIdx.x) * 40) + ((((int)threadIdx.x) / 5) * 20)) + (k_outer_outer * 5)) + (((int)threadIdx.x) % 5))];
    }
    __syncthreads();
    T_batch_matmul_NN_local[0] = (T_batch_matmul_NN_local[0] + (fmodf(ph_0[(((((int)blockIdx.x) * 40) + ((((int)threadIdx.x) / 5) * 4)) + k_outer_outer)], ph_3[(((((int)blockIdx.x) * 40) + ((((int)threadIdx.x) / 5) * 4)) + k_outer_outer)]) * ph_10_shared[(((((int)threadIdx.x) / 25) * 5) + (((int)threadIdx.x) % 5))]));
  }
  T_batch_matmul_NN[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = T_batch_matmul_NN_local[0];
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_add, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_add[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ((ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] - (ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] / ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])) + ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinhf((ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] - (ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] / ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
}


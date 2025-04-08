extern "C" __global__ void __launch_bounds__(40) default_function_kernel_3(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN_local[8];
  __shared__ float ph_3_shared[40];
  for (int b_c_outer_inner_init = 0; b_c_outer_inner_init < 2; ++b_c_outer_inner_init) {
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
      T_batch_matmul_NN_local[((b_c_outer_inner_init * 4) + i_c_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 5; ++k_outer_outer) {
    __syncthreads();
    ph_3_shared[((int)threadIdx.x)] = ph_3[((((((int)threadIdx.x) / 5) * 25) + (k_outer_outer * 5)) + (((int)threadIdx.x) % 5))];
    __syncthreads();
    for (int b_c_outer_inner = 0; b_c_outer_inner < 2; ++b_c_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
        T_batch_matmul_NN_local[((b_c_outer_inner * 4) + i_c_outer_inner)] = (T_batch_matmul_NN_local[((b_c_outer_inner * 4) + i_c_outer_inner)] + (ph_0[((((((((int)threadIdx.x) / 10) * 80) + (b_c_outer_inner * 40)) + (((((int)threadIdx.x) % 10) / 5) * 20)) + (i_c_outer_inner * 5)) + k_outer_outer)] * ph_3_shared[((((((int)threadIdx.x) / 10) * 10) + (b_c_outer_inner * 5)) + (((int)threadIdx.x) % 5))]));
      }
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    for (int i_inner = 0; i_inner < 4; ++i_inner) {
      T_batch_matmul_NN[((((((((int)threadIdx.x) / 10) * 80) + (b_inner * 40)) + (((((int)threadIdx.x) % 10) / 5) * 20)) + (i_inner * 5)) + (((int)threadIdx.x) % 5))] = T_batch_matmul_NN_local[((b_inner * 4) + i_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __cosf(atanhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinf(atanhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(4) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = atanf(ph_0[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_5(float* __restrict__ T_batch_matmul_NN, float* __restrict__ T_mod, float* __restrict__ ph_0) {
  T_mod[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = fmodf(T_batch_matmul_NN[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_4(float* __restrict__ T_batch_matmul_NN, float* __restrict__ compute) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = acosf(T_batch_matmul_NN[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}


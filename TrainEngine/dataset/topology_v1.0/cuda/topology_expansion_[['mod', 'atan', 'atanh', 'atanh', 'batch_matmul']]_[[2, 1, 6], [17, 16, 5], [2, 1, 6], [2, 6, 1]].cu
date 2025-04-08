extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_mod, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_mod[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = fmodf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(8) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = atanhf(atanhf(ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(8) default_function_kernel_3(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_8) {
  float T_batch_matmul_NN_local[4];
  __shared__ float ph_8_shared[8];
  for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
    T_batch_matmul_NN_local[i_c_inner_init] = 0.000000e+00f;
    T_batch_matmul_NN_local[(i_c_inner_init + 2)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 5; ++k_outer_outer) {
    __syncthreads();
    ph_8_shared[((int)threadIdx.x)] = ph_8[((((int)threadIdx.x) * 5) + k_outer_outer)];
    __syncthreads();
    for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
      T_batch_matmul_NN_local[i_c_inner] = (T_batch_matmul_NN_local[i_c_inner] + (atanhf(ph_0[((((((int)threadIdx.x) * 40) + (((int)blockIdx.x) * 20)) + (i_c_inner * 5)) + k_outer_outer)]) * ph_8_shared[((int)threadIdx.x)]));
      T_batch_matmul_NN_local[(i_c_inner + 2)] = (T_batch_matmul_NN_local[(i_c_inner + 2)] + (atanhf(ph_0[(((((((int)threadIdx.x) * 40) + (((int)blockIdx.x) * 20)) + (i_c_inner * 5)) + k_outer_outer) + 10)]) * ph_8_shared[((int)threadIdx.x)]));
    }
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    T_batch_matmul_NN[(((((int)threadIdx.x) * 8) + (((int)blockIdx.x) * 4)) + i_inner)] = T_batch_matmul_NN_local[i_inner];
    T_batch_matmul_NN[((((((int)threadIdx.x) * 8) + (((int)blockIdx.x) * 4)) + i_inner) + 2)] = T_batch_matmul_NN_local[(i_inner + 2)];
  }
}

extern "C" __global__ void __launch_bounds__(8) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = atanf(ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))]);
}


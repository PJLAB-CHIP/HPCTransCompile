extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_2(float* __restrict__ T_multiply, float* __restrict__ ph_0, float* __restrict__ ph_5) {
  T_multiply[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] * ph_5[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(80) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN_local[4];
  __shared__ float ph_3_shared[40];
  for (int b_c_inner_init = 0; b_c_inner_init < 4; ++b_c_inner_init) {
    T_batch_matmul_NN_local[b_c_inner_init] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 5; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 40) {
      ph_3_shared[((int)threadIdx.x)] = ph_3[((((((int)threadIdx.x) / 5) * 25) + (k_outer_outer * 5)) + (((int)threadIdx.x) % 5))];
    }
    __syncthreads();
    for (int b_c_inner = 0; b_c_inner < 4; ++b_c_inner) {
      T_batch_matmul_NN_local[b_c_inner] = (T_batch_matmul_NN_local[b_c_inner] + (ph_0[(((((((int)threadIdx.x) / 40) * 160) + (b_c_inner * 40)) + (((((int)threadIdx.x) % 40) / 5) * 5)) + k_outer_outer)] * ph_3_shared[((((((int)threadIdx.x) / 40) * 20) + (b_c_inner * 5)) + (((int)threadIdx.x) % 5))]));
    }
  }
  for (int b_inner = 0; b_inner < 4; ++b_inner) {
    T_batch_matmul_NN[((((((int)threadIdx.x) / 40) * 160) + (b_inner * 40)) + (((int)threadIdx.x) % 40))] = T_batch_matmul_NN_local[b_inner];
  }
}


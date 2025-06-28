extern "C" __global__ void default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((int)blockIdx.x)] = __expf(ph_0[((int)blockIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = acoshf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_3(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = acosf(atanhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(2) default_function_kernel_4(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = acosf(atanhf(ph_0[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(48) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0) {
  float T_batch_matmul_NN_local[2];
  __shared__ float compute_shared[48];
  for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
    T_batch_matmul_NN_local[j_c_inner_init] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    compute_shared[((int)threadIdx.x)] = ceilf(ph_0[(((((((int)blockIdx.x) >> 1) * 192) + ((((int)threadIdx.x) >> 1) * 8)) + (k_outer_outer * 2)) + (((int)threadIdx.x) & 1))]);
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
        T_batch_matmul_NN_local[j_c_inner] = (T_batch_matmul_NN_local[j_c_inner] + (compute_shared[(((((int)threadIdx.x) >> 1) * 2) + k_outer_inner)] * ph_0[((((((((((int)blockIdx.x) >> 1) * 192) + ((((int)threadIdx.x) >> 4) * 64)) + (k_outer_outer * 16)) + (k_outer_inner * 8)) + ((((int)blockIdx.x) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + j_c_inner)]));
      }
    }
  }
  for (int j_inner = 0; j_inner < 2; ++j_inner) {
    T_batch_matmul_NN[((((((((int)blockIdx.x) >> 1) * 192) + ((((int)threadIdx.x) >> 1) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner)] = T_batch_matmul_NN_local[j_inner];
  }
}


extern "C" __global__ void __launch_bounds__(32) default_function_kernel_3(float* __restrict__ T_multiply, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 9) {
    T_multiply[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (__expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) * ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN_local[9];
  __shared__ float ph_3_shared[4];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 3; ++i_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 3; ++i_c_inner_init) {
      T_batch_matmul_NN_local[((i_c_outer_inner_init * 3) + i_c_inner_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      ph_3_shared[ax0_ax1_fused_ax2_fused_outer_outer] = ph_3[(((k_outer_outer * 20) + (ax0_ax1_fused_ax2_fused_outer_outer * 5)) + ((int)blockIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 3; ++i_c_outer_inner) {
        for (int i_c_inner = 0; i_c_inner < 3; ++i_c_inner) {
          T_batch_matmul_NN_local[((i_c_outer_inner * 3) + i_c_inner)] = (T_batch_matmul_NN_local[((i_c_outer_inner * 3) + i_c_inner)] + (ph_0[((((i_c_outer_inner * 24) + (i_c_inner * 8)) + (k_outer_outer * 4)) + k_outer_inner)] * ph_3_shared[k_outer_inner]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 9; ++i_inner) {
    T_batch_matmul_NN[((i_inner * 5) + ((int)blockIdx.x))] = T_batch_matmul_NN_local[i_inner];
  }
}

extern "C" __global__ void __launch_bounds__(8) default_function_kernel_4(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = __expf(__expf(ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = acosf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = __cosf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
  }
}


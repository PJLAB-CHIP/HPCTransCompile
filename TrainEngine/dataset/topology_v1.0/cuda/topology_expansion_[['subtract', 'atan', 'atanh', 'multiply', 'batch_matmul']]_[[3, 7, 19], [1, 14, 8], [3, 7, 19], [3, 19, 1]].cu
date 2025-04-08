extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ T_subtract, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_subtract[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] - ph_3[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ T_multiply, float* __restrict__ ph_0) {
  T_multiply[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (atanhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) * ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(4) default_function_kernel_3(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_8) {
  float T_batch_matmul_NN_local[8];
  __shared__ float ph_8_shared[40];
  for (int b_c_inner_init = 0; b_c_inner_init < 4; ++b_c_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
      T_batch_matmul_NN_local[((b_c_inner_init * 2) + i_c_inner_init)] = 0.000000e+00f;
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_8_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 4) + ((int)threadIdx.x))] = ph_8[((ax0_ax1_fused_ax2_fused_outer_outer * 4) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_inner = 0; k_inner < 5; ++k_inner) {
    for (int b_c_inner = 0; b_c_inner < 4; ++b_c_inner) {
      for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
        T_batch_matmul_NN_local[((b_c_inner * 2) + i_c_inner)] = (T_batch_matmul_NN_local[((b_c_inner * 2) + i_c_inner)] + (atanhf(ph_0[(((((((((int)threadIdx.x) >> 1) * 160) + (b_c_inner * 40)) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 10)) + (i_c_inner * 5)) + k_inner)]) * ph_8_shared[((((((int)threadIdx.x) >> 1) * 20) + (b_c_inner * 5)) + k_inner)]));
      }
    }
  }
  for (int b_inner = 0; b_inner < 4; ++b_inner) {
    for (int i_inner = 0; i_inner < 2; ++i_inner) {
      T_batch_matmul_NN[((((((((int)threadIdx.x) >> 1) * 32) + (b_inner * 8)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + i_inner)] = T_batch_matmul_NN_local[((b_inner * 2) + i_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = atanf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}


extern "C" __global__ void __launch_bounds__(2) default_function_kernel_3(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_8) {
  float T_batch_matmul_NN_local[8];
  __shared__ float ph_8_shared[10];
  for (int b_c_inner_init = 0; b_c_inner_init < 2; ++b_c_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
      T_batch_matmul_NN_local[((b_c_inner_init * 2) + i_c_inner_init)] = 0.000000e+00f;
      T_batch_matmul_NN_local[(((b_c_inner_init * 2) + i_c_inner_init) + 4)] = 0.000000e+00f;
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_8_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 2) + ((int)threadIdx.x))] = ph_8[(((((int)blockIdx.x) * 10) + (ax0_ax1_fused_ax2_fused_outer_outer * 2)) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    for (int b_c_inner = 0; b_c_inner < 2; ++b_c_inner) {
      for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
        T_batch_matmul_NN_local[((b_c_inner * 2) + i_c_inner)] = (T_batch_matmul_NN_local[((b_c_inner * 2) + i_c_inner)] + (ceilf(ph_0[(((((((int)blockIdx.x) * 80) + (b_c_inner * 40)) + (((int)threadIdx.x) * 10)) + (i_c_inner * 5)) + k_outer_inner)]) * ph_8_shared[((b_c_inner * 5) + k_outer_inner)]));
        T_batch_matmul_NN_local[(((b_c_inner * 2) + i_c_inner) + 4)] = (T_batch_matmul_NN_local[(((b_c_inner * 2) + i_c_inner) + 4)] + (ceilf(ph_0[((((((((int)blockIdx.x) * 80) + (b_c_inner * 40)) + (((int)threadIdx.x) * 10)) + (i_c_inner * 5)) + k_outer_inner) + 20)]) * ph_8_shared[((b_c_inner * 5) + k_outer_inner)]));
      }
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    for (int i_inner = 0; i_inner < 2; ++i_inner) {
      T_batch_matmul_NN[((((((int)blockIdx.x) * 16) + (b_inner * 8)) + (((int)threadIdx.x) * 2)) + i_inner)] = T_batch_matmul_NN_local[((b_inner * 2) + i_inner)];
      T_batch_matmul_NN[(((((((int)blockIdx.x) * 16) + (b_inner * 8)) + (((int)threadIdx.x) * 2)) + i_inner) + 4)] = T_batch_matmul_NN_local[(((b_inner * 2) + i_inner) + 4)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_divide, float* __restrict__ ph_0) {
  T_divide[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] / fabsf(ceilf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
}

extern "C" __global__ void __launch_bounds__(8) default_function_kernel_2(float* __restrict__ T_mod, float* __restrict__ ph_0) {
  T_mod[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = fmodf(ceilf(ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))]), ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = acosf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}


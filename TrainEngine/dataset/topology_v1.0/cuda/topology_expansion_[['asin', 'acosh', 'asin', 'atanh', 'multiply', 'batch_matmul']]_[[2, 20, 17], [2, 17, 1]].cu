extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinf(acoshf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(2) default_function_kernel_2(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_7) {
  float T_batch_matmul_NN_local[8];
  __shared__ float T_multiply_shared[80];
  __shared__ float ph_7_shared[10];
  for (int b_c_outer_inner_init = 0; b_c_outer_inner_init < 2; ++b_c_outer_inner_init) {
    for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
      for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
        T_batch_matmul_NN_local[(((b_c_outer_inner_init * 4) + (i_c_outer_inner_init * 2)) + i_c_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    for (int ax0_ax1_fused_ax2_fused_inner_s = 0; ax0_ax1_fused_ax2_fused_inner_s < 4; ++ax0_ax1_fused_ax2_fused_inner_s) {
      T_multiply_shared[(((ax0_ax1_fused_ax2_fused_outer_outer * 8) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_inner_s)] = (ph_0[((((((int)blockIdx.x) * 80) + (ax0_ax1_fused_ax2_fused_outer_outer * 8)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_inner_s)] * atanhf(ph_0[((((((int)blockIdx.x) * 80) + (ax0_ax1_fused_ax2_fused_outer_outer * 8)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_inner_s)]));
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_outer_outer_1 < 5; ++ax0_ax1_fused_ax2_fused_outer_outer_1) {
    ph_7_shared[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 2) + ((int)threadIdx.x))] = ph_7[(((((int)blockIdx.x) * 10) + (ax0_ax1_fused_ax2_fused_outer_outer_1 * 2)) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int b_c_outer_inner = 0; b_c_outer_inner < 2; ++b_c_outer_inner) {
    for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
      for (int k_inner = 0; k_inner < 5; ++k_inner) {
        for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
          T_batch_matmul_NN_local[(((b_c_outer_inner * 4) + (i_c_outer_inner * 2)) + i_c_inner)] = (T_batch_matmul_NN_local[(((b_c_outer_inner * 4) + (i_c_outer_inner * 2)) + i_c_inner)] + (T_multiply_shared[(((((b_c_outer_inner * 40) + (((int)threadIdx.x) * 20)) + (i_c_outer_inner * 10)) + (i_c_inner * 5)) + k_inner)] * ph_7_shared[((b_c_outer_inner * 5) + k_inner)]));
        }
      }
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    for (int i_inner = 0; i_inner < 4; ++i_inner) {
      T_batch_matmul_NN[((((((int)blockIdx.x) * 16) + (b_inner * 8)) + (((int)threadIdx.x) * 4)) + i_inner)] = T_batch_matmul_NN_local[((b_inner * 4) + i_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}


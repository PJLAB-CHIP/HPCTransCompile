extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_4) {
  float T_batch_matmul_NN_local[16];
  __shared__ float compute_shared[80];
  __shared__ float ph_4_shared[10];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int b_c_inner_init = 0; b_c_inner_init < 2; ++b_c_inner_init) {
      for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
        T_batch_matmul_NN_local[(((b_c_inner_init * 8) + (i_c_outer_inner_init * 2)) + i_c_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 80; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    compute_shared[ax0_ax1_fused_ax2_fused_outer_outer] = fabsf(ph_0[((((int)blockIdx.x) * 80) + ax0_ax1_fused_ax2_fused_outer_outer)]);
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_outer_outer_1 < 10; ++ax0_ax1_fused_ax2_fused_outer_outer_1) {
    ph_4_shared[ax0_ax1_fused_ax2_fused_outer_outer_1] = ph_4[((((int)blockIdx.x) * 10) + ax0_ax1_fused_ax2_fused_outer_outer_1)];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
      for (int b_c_inner = 0; b_c_inner < 2; ++b_c_inner) {
        for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
          T_batch_matmul_NN_local[(((b_c_inner * 8) + (i_c_outer_inner * 2)) + i_c_inner)] = (T_batch_matmul_NN_local[(((b_c_inner * 8) + (i_c_outer_inner * 2)) + i_c_inner)] + (compute_shared[((((b_c_inner * 40) + (i_c_outer_inner * 10)) + (i_c_inner * 5)) + k_outer_inner)] * ph_4_shared[((b_c_inner * 5) + k_outer_inner)]));
        }
      }
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    for (int i_inner = 0; i_inner < 8; ++i_inner) {
      T_batch_matmul_NN[(((((int)blockIdx.x) * 16) + (b_inner * 8)) + i_inner)] = T_batch_matmul_NN_local[((b_inner * 8) + i_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}


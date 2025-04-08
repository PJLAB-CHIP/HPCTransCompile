extern "C" __global__ void __launch_bounds__(2) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_4) {
  float T_batch_matmul_NN_local[8];
  __shared__ float compute_shared[80];
  __shared__ float ph_4_shared[40];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    for (int b_c_inner_init = 0; b_c_inner_init < 4; ++b_c_inner_init) {
      T_batch_matmul_NN_local[((b_c_inner_init * 2) + i_c_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 40; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    compute_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 2) + ((int)threadIdx.x))] = asinhf(ph_0[(((((ax0_ax1_fused_ax2_fused_outer_outer / 5) * 40) + (((int)blockIdx.x) * 10)) + ((ax0_ax1_fused_ax2_fused_outer_outer % 5) * 2)) + ((int)threadIdx.x))]);
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_outer_outer_1 < 20; ++ax0_ax1_fused_ax2_fused_outer_outer_1) {
    ph_4_shared[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 2) + ((int)threadIdx.x))] = ph_4[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 2) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
    for (int k_inner = 0; k_inner < 5; ++k_inner) {
      for (int b_c_inner = 0; b_c_inner < 4; ++b_c_inner) {
        T_batch_matmul_NN_local[((b_c_inner * 2) + i_c_outer_inner)] = (T_batch_matmul_NN_local[((b_c_inner * 2) + i_c_outer_inner)] + (compute_shared[((((((int)threadIdx.x) * 40) + (b_c_inner * 10)) + (i_c_outer_inner * 5)) + k_inner)] * ph_4_shared[(((((int)threadIdx.x) * 20) + (b_c_inner * 5)) + k_inner)]));
      }
    }
  }
  for (int b_inner = 0; b_inner < 4; ++b_inner) {
    for (int i_inner = 0; i_inner < 2; ++i_inner) {
      T_batch_matmul_NN[((((((int)threadIdx.x) * 32) + (b_inner * 8)) + (((int)blockIdx.x) * 2)) + i_inner)] = T_batch_matmul_NN_local[((b_inner * 2) + i_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}


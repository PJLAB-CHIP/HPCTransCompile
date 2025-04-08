extern "C" __global__ void __launch_bounds__(72) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  float T_batch_matmul_NN[8];
  __shared__ float compute_shared[576];
  for (int i_outer_inner_init = 0; i_outer_inner_init < 2; ++i_outer_inner_init) {
    for (int i_inner_init = 0; i_inner_init < 2; ++i_inner_init) {
      for (int j_inner_init = 0; j_inner_init < 2; ++j_inner_init) {
        T_batch_matmul_NN[(((i_outer_inner_init * 4) + (i_inner_init * 2)) + j_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    compute_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 72) + ((int)threadIdx.x))] = __sinf(ph_0[((ax0_ax1_fused_ax2_fused_outer_outer * 72) + ((int)threadIdx.x))]);
  }
  __syncthreads();
  for (int i_outer_inner = 0; i_outer_inner < 2; ++i_outer_inner) {
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      for (int i_inner = 0; i_inner < 2; ++i_inner) {
        for (int j_inner = 0; j_inner < 2; ++j_inner) {
          T_batch_matmul_NN[(((i_outer_inner * 4) + (i_inner * 2)) + j_inner)] = (T_batch_matmul_NN[(((i_outer_inner * 4) + (i_inner * 2)) + j_inner)] + (ph_0[(((((((int)threadIdx.x) >> 2) * 32) + (i_outer_inner * 16)) + (i_inner * 8)) + k_inner)] * compute_shared[(((((((int)threadIdx.x) >> 3) * 64) + (k_inner * 8)) + ((((int)threadIdx.x) & 3) * 2)) + j_inner)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
      compute[(((((((int)threadIdx.x) >> 2) * 32) + (i1_inner * 8)) + ((((int)threadIdx.x) & 3) * 2)) + i2_inner)] = __sinf(T_batch_matmul_NN[((i1_inner * 2) + i2_inner)]);
    }
  }
}


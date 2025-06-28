void default_function_kernel(float* T_batch_matmul_NN, float* compute, float* ph_0, float* ph_4) {
  float auto_scheduler_layout_transform[4];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 20; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 2; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax4 = 0; ax4 < 2; ++ax4) {
      auto_scheduler_layout_transform[((ax0_ax1_fused_ax2_fused * 2) + ax4)] = ph_4[((ax0_ax1_fused_ax2_fused * 2) + ax4)];
    }
  }
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused = 0; b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused < 2; ++b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused) {
    for (int32_t i_outer_inner_init = 0; i_outer_inner_init < 5; ++i_outer_inner_init) {
      T_batch_matmul_NN[((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 5) + i_outer_inner_init)] = 0.000000e+00f;
    }
    for (int32_t k_outer = 0; k_outer < 2; ++k_outer) {
      for (int32_t i_outer_inner = 0; i_outer_inner < 5; ++i_outer_inner) {
        T_batch_matmul_NN[((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 5) + i_outer_inner)] = (T_batch_matmul_NN[((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 5) + i_outer_inner)] + (asinhf(ph_0[(((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 10) + (i_outer_inner * 2)) + k_outer)]) * auto_scheduler_layout_transform[((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 2) + k_outer)]));
      }
    }
  }
}


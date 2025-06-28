void default_function_kernel(float* T_batch_matmul_NN, float* T_multiply, float* T_multiply_1, float* compute, float* ph_0, float* ph_8) {
  float auto_scheduler_layout_transform[18];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 144; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = fabsf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 144; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = (acoshf(ph_0[ax0_ax1_fused_ax2_fused]) * ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 144; ++ax0_ax1_fused_ax2_fused_1) {
    T_multiply_1[ax0_ax1_fused_ax2_fused_1] = (fmodf(ph_0[ax0_ax1_fused_ax2_fused_1], asinf(ph_0[ax0_ax1_fused_ax2_fused_1])) * ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
  for (int32_t ax5 = 0; ax5 < 6; ++ax5) {
    for (int32_t ax7 = 0; ax7 < 3; ++ax7) {
      auto_scheduler_layout_transform[((ax5 * 3) + ax7)] = ph_8[((ax5 * 3) + ax7)];
    }
  }
  for (int32_t b_outer_inner_init = 0; b_outer_inner_init < 6; ++b_outer_inner_init) {
    for (int32_t i_outer_inner_init = 0; i_outer_inner_init < 2; ++i_outer_inner_init) {
      for (int32_t i_inner_init = 0; i_inner_init < 4; ++i_inner_init) {
        T_batch_matmul_NN[(((b_outer_inner_init * 8) + (i_outer_inner_init * 4)) + i_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int32_t b_outer_inner = 0; b_outer_inner < 6; ++b_outer_inner) {
    for (int32_t i_outer_inner = 0; i_outer_inner < 2; ++i_outer_inner) {
      for (int32_t k_inner = 0; k_inner < 3; ++k_inner) {
        for (int32_t i_inner = 0; i_inner < 4; ++i_inner) {
          T_batch_matmul_NN[(((b_outer_inner * 8) + (i_outer_inner * 4)) + i_inner)] = (T_batch_matmul_NN[(((b_outer_inner * 8) + (i_outer_inner * 4)) + i_inner)] + (fmodf(ph_0[((((b_outer_inner * 24) + (i_outer_inner * 12)) + (i_inner * 3)) + k_inner)], asinf(ph_0[((((b_outer_inner * 24) + (i_outer_inner * 12)) + (i_inner * 3)) + k_inner)])) * auto_scheduler_layout_transform[((b_outer_inner * 3) + k_inner)]));
        }
      }
    }
  }
}


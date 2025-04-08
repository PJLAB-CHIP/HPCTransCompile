void default_function_kernel(float* T_batch_matmul_NN, float* T_multiply, float* compute, float* ph_0, float* ph_3, float* ph_5) {
  float auto_scheduler_layout_transform[1805];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 722; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  for (int32_t ax4 = 0; ax4 < 19; ++ax4) {
    for (int32_t ax6 = 0; ax6 < 5; ++ax6) {
      for (int32_t ax8 = 0; ax8 < 19; ++ax8) {
        auto_scheduler_layout_transform[(((ax4 * 95) + (ax6 * 19)) + ax8)] = ph_3[(((ax8 * 95) + (ax4 * 5)) + ax6)];
      }
    }
  }
  for (int32_t i_outer_inner_init = 0; i_outer_inner_init < 2; ++i_outer_inner_init) {
    for (int32_t j_outer_inner_init = 0; j_outer_inner_init < 5; ++j_outer_inner_init) {
      for (int32_t b_inner_init = 0; b_inner_init < 19; ++b_inner_init) {
        T_batch_matmul_NN[(((b_inner_init * 10) + (i_outer_inner_init * 5)) + j_outer_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int32_t k_outer = 0; k_outer < 19; ++k_outer) {
    for (int32_t i_outer_inner = 0; i_outer_inner < 2; ++i_outer_inner) {
      for (int32_t j_outer_inner = 0; j_outer_inner < 5; ++j_outer_inner) {
        for (int32_t b_inner = 0; b_inner < 19; ++b_inner) {
          T_batch_matmul_NN[(((b_inner * 10) + (i_outer_inner * 5)) + j_outer_inner)] = (T_batch_matmul_NN[(((b_inner * 10) + (i_outer_inner * 5)) + j_outer_inner)] + (ph_0[(((b_inner * 38) + (i_outer_inner * 19)) + k_outer)] * auto_scheduler_layout_transform[(((k_outer * 95) + (j_outer_inner * 19)) + b_inner)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 722; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] * ph_5[ax0_ax1_fused_ax2_fused]);
  }
}


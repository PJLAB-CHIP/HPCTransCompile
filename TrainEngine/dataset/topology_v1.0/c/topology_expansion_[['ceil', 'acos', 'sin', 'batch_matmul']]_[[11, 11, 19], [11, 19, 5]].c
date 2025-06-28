void default_function_kernel(float* T_batch_matmul_NN, float* compute, float* compute_1, float* ph_0, float* ph_5) {
  float auto_scheduler_layout_transform[1045];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 2299; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = ceilf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 2299; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = sinf(acosf(ph_0[i0_i1_fused_i2_fused_1]));
  }
  for (int32_t ax4 = 0; ax4 < 19; ++ax4) {
    for (int32_t ax6 = 0; ax6 < 5; ++ax6) {
      for (int32_t ax8 = 0; ax8 < 11; ++ax8) {
        auto_scheduler_layout_transform[(((ax4 * 55) + (ax6 * 11)) + ax8)] = ph_5[(((ax8 * 95) + (ax4 * 5)) + ax6)];
      }
    }
  }
  for (int32_t i_outer_inner_init = 0; i_outer_inner_init < 11; ++i_outer_inner_init) {
    for (int32_t j_outer_inner_init = 0; j_outer_inner_init < 5; ++j_outer_inner_init) {
      for (int32_t b_inner_init = 0; b_inner_init < 11; ++b_inner_init) {
        T_batch_matmul_NN[(((b_inner_init * 55) + (i_outer_inner_init * 5)) + j_outer_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int32_t k_outer = 0; k_outer < 19; ++k_outer) {
    for (int32_t i_outer_inner = 0; i_outer_inner < 11; ++i_outer_inner) {
      for (int32_t j_outer_inner = 0; j_outer_inner < 5; ++j_outer_inner) {
        for (int32_t b_inner = 0; b_inner < 11; ++b_inner) {
          T_batch_matmul_NN[(((b_inner * 55) + (i_outer_inner * 5)) + j_outer_inner)] = (T_batch_matmul_NN[(((b_inner * 55) + (i_outer_inner * 5)) + j_outer_inner)] + (ph_0[(((b_inner * 209) + (i_outer_inner * 19)) + k_outer)] * auto_scheduler_layout_transform[(((k_outer * 55) + (j_outer_inner * 11)) + b_inner)]));
        }
      }
    }
  }
}


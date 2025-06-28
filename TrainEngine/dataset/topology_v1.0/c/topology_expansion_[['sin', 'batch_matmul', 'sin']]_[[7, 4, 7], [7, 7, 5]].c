void default_function_kernel(float* T_batch_matmul_NN, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  float auto_scheduler_layout_transform[245];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 196; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = sinf(ph_0[i0_i1_fused_i2_fused]);
  }
  for (int32_t ax4 = 0; ax4 < 7; ++ax4) {
    for (int32_t ax5 = 0; ax5 < 7; ++ax5) {
      for (int32_t ax6 = 0; ax6 < 5; ++ax6) {
        auto_scheduler_layout_transform[(((ax4 * 35) + (ax5 * 5)) + ax6)] = ph_3[(((ax5 * 35) + (ax4 * 5)) + ax6)];
      }
    }
  }
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused = 0; b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused < 2; ++b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused) {
    for (int32_t b_outer_inner_init = 0; b_outer_inner_init < 7; ++b_outer_inner_init) {
      for (int32_t j_outer_inner_init = 0; j_outer_inner_init < 5; ++j_outer_inner_init) {
        for (int32_t i_inner_init = 0; i_inner_init < 2; ++i_inner_init) {
          T_batch_matmul_NN[((((b_outer_inner_init * 20) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 10)) + (i_inner_init * 5)) + j_outer_inner_init)] = 0.000000e+00f;
        }
      }
    }
    for (int32_t k_outer = 0; k_outer < 7; ++k_outer) {
      for (int32_t b_outer_inner = 0; b_outer_inner < 7; ++b_outer_inner) {
        for (int32_t j_outer_inner = 0; j_outer_inner < 5; ++j_outer_inner) {
          for (int32_t i_inner = 0; i_inner < 2; ++i_inner) {
            T_batch_matmul_NN[((((b_outer_inner * 20) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 10)) + (i_inner * 5)) + j_outer_inner)] = (T_batch_matmul_NN[((((b_outer_inner * 20) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 10)) + (i_inner * 5)) + j_outer_inner)] + (ph_0[((((b_outer_inner * 28) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 14)) + (i_inner * 7)) + k_outer)] * auto_scheduler_layout_transform[(((k_outer * 35) + (b_outer_inner * 5)) + j_outer_inner)]));
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 196; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = sinf(ph_0[i0_i1_fused_i2_fused_1]);
  }
}


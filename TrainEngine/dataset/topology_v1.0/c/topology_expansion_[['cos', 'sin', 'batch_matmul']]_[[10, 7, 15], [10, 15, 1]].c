void default_function_kernel(float* T_batch_matmul_NN, float* compute, float* ph_0, float* ph_4) {
  float auto_scheduler_layout_transform[150];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1050; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = cosf(ph_0[i0_i1_fused_i2_fused]);
  }
  for (int32_t ax5 = 0; ax5 < 2; ++ax5) {
    for (int32_t ax7 = 0; ax7 < 15; ++ax7) {
      for (int32_t ax8 = 0; ax8 < 5; ++ax8) {
        auto_scheduler_layout_transform[(((ax5 * 75) + (ax7 * 5)) + ax8)] = ph_4[(((ax5 * 75) + (ax8 * 15)) + ax7)];
      }
    }
  }
  for (int32_t b_outer_inner_init = 0; b_outer_inner_init < 2; ++b_outer_inner_init) {
    for (int32_t b_inner_init = 0; b_inner_init < 5; ++b_inner_init) {
      for (int32_t i_inner_init = 0; i_inner_init < 7; ++i_inner_init) {
        T_batch_matmul_NN[(((b_outer_inner_init * 35) + (b_inner_init * 7)) + i_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int32_t b_outer_inner = 0; b_outer_inner < 2; ++b_outer_inner) {
    for (int32_t k_inner = 0; k_inner < 15; ++k_inner) {
      for (int32_t b_inner = 0; b_inner < 5; ++b_inner) {
        for (int32_t i_inner = 0; i_inner < 7; ++i_inner) {
          T_batch_matmul_NN[(((b_outer_inner * 35) + (b_inner * 7)) + i_inner)] = (T_batch_matmul_NN[(((b_outer_inner * 35) + (b_inner * 7)) + i_inner)] + (sinf(ph_0[((((b_outer_inner * 525) + (b_inner * 105)) + (i_inner * 15)) + k_inner)]) * auto_scheduler_layout_transform[(((b_outer_inner * 75) + (k_inner * 5)) + b_inner)]));
        }
      }
    }
  }
}


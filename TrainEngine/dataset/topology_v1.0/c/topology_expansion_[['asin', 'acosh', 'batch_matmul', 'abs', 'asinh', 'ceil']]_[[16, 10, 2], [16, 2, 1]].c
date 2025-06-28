void default_function_kernel(float* T_batch_matmul_NN, float* compute, float* compute_1, float* compute_2, float* ph_0, float* ph_4) {
  float auto_scheduler_layout_transform[32];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 320; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf(ph_0[i0_i1_fused_i2_fused]);
  }
  for (int32_t ax5 = 0; ax5 < 16; ++ax5) {
    for (int32_t ax7 = 0; ax7 < 2; ++ax7) {
      auto_scheduler_layout_transform[((ax5 * 2) + ax7)] = ph_4[((ax5 * 2) + ax7)];
    }
  }
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused = 0; b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused < 2; ++b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused) {
    for (int32_t b_outer_inner_init = 0; b_outer_inner_init < 16; ++b_outer_inner_init) {
      for (int32_t i_inner_init = 0; i_inner_init < 5; ++i_inner_init) {
        T_batch_matmul_NN[(((b_outer_inner_init * 10) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 5)) + i_inner_init)] = 0.000000e+00f;
      }
    }
    for (int32_t b_outer_inner = 0; b_outer_inner < 16; ++b_outer_inner) {
      for (int32_t k_inner = 0; k_inner < 2; ++k_inner) {
        for (int32_t i_inner = 0; i_inner < 5; ++i_inner) {
          T_batch_matmul_NN[(((b_outer_inner * 10) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 5)) + i_inner)] = (T_batch_matmul_NN[(((b_outer_inner * 10) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 5)) + i_inner)] + (acoshf(ph_0[((((b_outer_inner * 20) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 10)) + (i_inner * 2)) + k_inner)]) * auto_scheduler_layout_transform[((b_outer_inner * 2) + k_inner)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 320; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = asinhf(fabsf(ph_0[i0_i1_fused_i2_fused_1]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 320; ++i0_i1_fused_i2_fused_2) {
    compute_2[i0_i1_fused_i2_fused_2] = ceilf(fabsf(ph_0[i0_i1_fused_i2_fused_2]));
  }
}


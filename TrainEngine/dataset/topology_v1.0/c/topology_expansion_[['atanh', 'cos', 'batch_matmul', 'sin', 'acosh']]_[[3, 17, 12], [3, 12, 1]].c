void default_function_kernel(float* T_batch_matmul_NN, float* compute, float* compute_1, float* ph_0, float* ph_4) {
  float auto_scheduler_layout_transform[36];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 612; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = atanhf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 3; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax7 = 0; ax7 < 12; ++ax7) {
      auto_scheduler_layout_transform[((ax0_ax1_fused_ax2_fused * 12) + ax7)] = ph_4[((ax0_ax1_fused_ax2_fused * 12) + ax7)];
    }
  }
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused = 0; b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused < 3; ++b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused) {
    for (int32_t i_inner_init = 0; i_inner_init < 17; ++i_inner_init) {
      T_batch_matmul_NN[((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 17) + i_inner_init)] = 0.000000e+00f;
    }
    for (int32_t k_inner = 0; k_inner < 12; ++k_inner) {
      for (int32_t i_inner = 0; i_inner < 17; ++i_inner) {
        T_batch_matmul_NN[((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 17) + i_inner)] = (T_batch_matmul_NN[((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 17) + i_inner)] + (cosf(ph_0[(((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 204) + (i_inner * 12)) + k_inner)]) * auto_scheduler_layout_transform[((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 12) + k_inner)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 612; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = acoshf(sinf(ph_0[i0_i1_fused_i2_fused_1]));
  }
}


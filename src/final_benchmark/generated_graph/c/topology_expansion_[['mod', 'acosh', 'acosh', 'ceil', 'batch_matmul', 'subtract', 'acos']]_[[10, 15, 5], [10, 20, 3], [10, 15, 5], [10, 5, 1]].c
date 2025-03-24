void default_function_kernel(float* T_batch_matmul_NN, float* T_subtract, float* compute, float* compute_1, float* compute_2, float* ph_0, float* ph_3, float* ph_8) {
  float auto_scheduler_layout_transform[50];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 750; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = acoshf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 750; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = ceilf(acoshf(ph_0[i0_i1_fused_i2_fused_1]));
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 2; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax4 = 0; ax4 < 5; ++ax4) {
      for (int32_t ax8 = 0; ax8 < 5; ++ax8) {
        auto_scheduler_layout_transform[(((ax0_ax1_fused_ax2_fused * 25) + (ax4 * 5)) + ax8)] = ph_8[(((ax0_ax1_fused_ax2_fused * 25) + (ax8 * 5)) + ax4)];
      }
    }
  }
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused = 0; b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused < 2; ++b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused) {
    for (int32_t i_outer_inner_init = 0; i_outer_inner_init < 3; ++i_outer_inner_init) {
      for (int32_t b_inner_init = 0; b_inner_init < 5; ++b_inner_init) {
        for (int32_t i_inner_init = 0; i_inner_init < 5; ++i_inner_init) {
          T_batch_matmul_NN[((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 75) + (b_inner_init * 15)) + (i_outer_inner_init * 5)) + i_inner_init)] = 0.000000e+00f;
        }
      }
    }
    for (int32_t k_outer = 0; k_outer < 5; ++k_outer) {
      for (int32_t i_outer_inner = 0; i_outer_inner < 3; ++i_outer_inner) {
        for (int32_t b_inner = 0; b_inner < 5; ++b_inner) {
          for (int32_t i_inner = 0; i_inner < 5; ++i_inner) {
            T_batch_matmul_NN[((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 75) + (b_inner * 15)) + (i_outer_inner * 5)) + i_inner)] = (T_batch_matmul_NN[((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 75) + (b_inner * 15)) + (i_outer_inner * 5)) + i_inner)] + (acoshf(ph_0[(((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 375) + (b_inner * 75)) + (i_outer_inner * 25)) + (i_inner * 5)) + k_outer)]) * auto_scheduler_layout_transform[(((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 25) + (k_outer * 5)) + b_inner)]));
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 750; ++ax0_ax1_fused_ax2_fused_1) {
    T_subtract[ax0_ax1_fused_ax2_fused_1] = (fmodf(ph_0[ax0_ax1_fused_ax2_fused_1], ph_3[ax0_ax1_fused_ax2_fused_1]) - ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 750; ++i0_i1_fused_i2_fused_2) {
    compute_2[i0_i1_fused_i2_fused_2] = acosf(fmodf(ph_0[i0_i1_fused_i2_fused_2], ph_3[i0_i1_fused_i2_fused_2]));
  }
}


void default_function_kernel(float* T_mod, float* compute, float* compute_1, float* compute_2, float* compute_3, float* ph_0, float* ph_10, float* ph_3) {
  float auto_scheduler_layout_transform[1040];
  float T_batch_matmul_NN[325];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1040; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = atanf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 1040; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = asinf(atanhf(ph_0[i0_i1_fused_i2_fused_1]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 1040; ++i0_i1_fused_i2_fused_2) {
    compute_2[i0_i1_fused_i2_fused_2] = cosf(atanhf(ph_0[i0_i1_fused_i2_fused_2]));
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 5; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
      for (int32_t ax7 = 0; ax7 < 4; ++ax7) {
        for (int32_t ax8 = 0; ax8 < 13; ++ax8) {
          auto_scheduler_layout_transform[((((ax0_ax1_fused_ax2_fused * 208) + (ax4 * 52)) + (ax7 * 13)) + ax8)] = ph_3[((((ax8 * 80) + (ax4 * 20)) + (ax7 * 5)) + ax0_ax1_fused_ax2_fused)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused = 0; b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused < 5; ++b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused) {
    for (int32_t b_inner_init = 0; b_inner_init < 13; ++b_inner_init) {
      for (int32_t i_inner_init = 0; i_inner_init < 5; ++i_inner_init) {
        T_batch_matmul_NN[(((b_inner_init * 25) + (i_inner_init * 5)) + b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused)] = 0.000000e+00f;
      }
    }
    for (int32_t k_outer = 0; k_outer < 4; ++k_outer) {
      for (int32_t k_inner = 0; k_inner < 4; ++k_inner) {
        for (int32_t b_inner = 0; b_inner < 13; ++b_inner) {
          for (int32_t i_inner = 0; i_inner < 5; ++i_inner) {
            T_batch_matmul_NN[(((b_inner * 25) + (i_inner * 5)) + b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused)] = (T_batch_matmul_NN[(((b_inner * 25) + (i_inner * 5)) + b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused)] + (ph_0[((((b_inner * 80) + (i_inner * 16)) + (k_outer * 4)) + k_inner)] * auto_scheduler_layout_transform[((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused * 208) + (k_outer * 52)) + (k_inner * 13)) + b_inner)]));
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_3 = 0; i0_i1_fused_i2_fused_3 < 325; ++i0_i1_fused_i2_fused_3) {
    compute_3[i0_i1_fused_i2_fused_3] = acosf(T_batch_matmul_NN[i0_i1_fused_i2_fused_3]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 325; ++ax0_ax1_fused_ax2_fused_1) {
    T_mod[ax0_ax1_fused_ax2_fused_1] = fmodf(T_batch_matmul_NN[ax0_ax1_fused_ax2_fused_1], ph_10[ax0_ax1_fused_ax2_fused_1]);
  }
}


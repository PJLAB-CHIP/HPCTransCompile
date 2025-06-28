void default_function_kernel(float* T_add, float* T_batch_matmul_NN, float* T_multiply, float* compute, float* ph_0, float* ph_3, float* ph_7) {
  float auto_scheduler_layout_transform[60];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 480; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 3; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax4 = 0; ax4 < 10; ++ax4) {
      for (int32_t ax5 = 0; ax5 < 2; ++ax5) {
        auto_scheduler_layout_transform[(((ax0_ax1_fused_ax2_fused * 20) + (ax4 * 2)) + ax5)] = ph_7[(((ax0_ax1_fused_ax2_fused * 20) + (ax5 * 10)) + ax4)];
      }
    }
  }
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused = 0; b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused < 24; ++b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused) {
    for (int32_t b_outer_inner_init = 0; b_outer_inner_init < 2; ++b_outer_inner_init) {
      T_batch_matmul_NN[((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused >> 3) * 16) + (b_outer_inner_init * 8)) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused & 7))] = 0.000000e+00f;
    }
    for (int32_t k_outer = 0; k_outer < 10; ++k_outer) {
      for (int32_t b_outer_inner = 0; b_outer_inner < 2; ++b_outer_inner) {
        T_batch_matmul_NN[((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused >> 3) * 16) + (b_outer_inner * 8)) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused & 7))] = (T_batch_matmul_NN[((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused >> 3) * 16) + (b_outer_inner * 8)) + (b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused & 7))] + (acosf(ph_0[(((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused >> 3) * 160) + (b_outer_inner * 80)) + ((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused & 7) * 10)) + k_outer)]) * auto_scheduler_layout_transform[((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused >> 3) * 20) + (k_outer * 2)) + b_outer_inner)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 480; ++ax0_ax1_fused_ax2_fused_1) {
    T_multiply[ax0_ax1_fused_ax2_fused_1] = (fmodf(ph_0[ax0_ax1_fused_ax2_fused_1], ph_3[ax0_ax1_fused_ax2_fused_1]) * ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_2 = 0; ax0_ax1_fused_ax2_fused_2 < 480; ++ax0_ax1_fused_ax2_fused_2) {
    T_add[ax0_ax1_fused_ax2_fused_2] = (fmodf(ph_0[ax0_ax1_fused_ax2_fused_2], ph_3[ax0_ax1_fused_ax2_fused_2]) + ph_0[ax0_ax1_fused_ax2_fused_2]);
  }
}


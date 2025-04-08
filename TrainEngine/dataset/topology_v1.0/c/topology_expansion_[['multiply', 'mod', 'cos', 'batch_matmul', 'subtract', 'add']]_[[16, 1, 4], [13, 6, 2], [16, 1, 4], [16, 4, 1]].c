void default_function_kernel(float* T_add, float* T_batch_matmul_NN, float* T_mod, float* T_subtract, float* ph_0, float* ph_3, float* ph_7) {
  float auto_scheduler_layout_transform[64];
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 16; ++ax0) {
    for (int32_t ax2 = 0; ax2 < 4; ++ax2) {
      T_mod[((ax0 * 4) + ax2)] = fmodf(ph_0[((ax0 * 4) + ax2)], ph_3[((ax0 * 4) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 16; ++ax0_ax1_fused) {
    for (int32_t ax2_1 = 0; ax2_1 < 4; ++ax2_1) {
      auto_scheduler_layout_transform[((ax0_ax1_fused * 4) + ax2_1)] = ph_7[((ax0_ax1_fused * 4) + ax2_1)];
    }
  }
  #pragma omp parallel for
  for (int32_t b_i_fused_j_fused = 0; b_i_fused_j_fused < 16; ++b_i_fused_j_fused) {
    T_batch_matmul_NN[b_i_fused_j_fused] = 0.000000e+00f;
    for (int32_t k = 0; k < 4; ++k) {
      T_batch_matmul_NN[b_i_fused_j_fused] = (T_batch_matmul_NN[b_i_fused_j_fused] + (cosf(ph_0[((b_i_fused_j_fused * 4) + k)]) * auto_scheduler_layout_transform[((b_i_fused_j_fused * 4) + k)]));
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 64; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = ((ph_0[ax0_ax1_fused_ax2_fused] * ph_3[ax0_ax1_fused_ax2_fused]) - ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 64; ++ax0_ax1_fused_ax2_fused_1) {
    T_add[ax0_ax1_fused_ax2_fused_1] = ((ph_0[ax0_ax1_fused_ax2_fused_1] * ph_3[ax0_ax1_fused_ax2_fused_1]) + ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
}


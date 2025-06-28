void default_function_kernel(float* T_mod, float* T_mod_1, float* T_multiply, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 196; ++ax0_ax1_fused_ax2_fused) {
    T_mod[ax0_ax1_fused_ax2_fused] = fmodf(ph_0[ax0_ax1_fused_ax2_fused], ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 28; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 7; ++ax2) {
      T_mod_1[((ax0_ax1_fused * 7) + ax2)] = fmodf((ph_0[((ax0_ax1_fused * 7) + ax2)] - ceilf(ph_0[((ax0_ax1_fused * 7) + ax2)])), ph_0[((ax0_ax1_fused * 7) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < 28; ++ax0_ax1_fused_1) {
    for (int32_t ax2_1 = 0; ax2_1 < 7; ++ax2_1) {
      T_multiply[((ax0_ax1_fused_1 * 7) + ax2_1)] = ((ph_0[((ax0_ax1_fused_1 * 7) + ax2_1)] - ceilf(ph_0[((ax0_ax1_fused_1 * 7) + ax2_1)])) * ph_0[((ax0_ax1_fused_1 * 7) + ax2_1)]);
    }
  }
}


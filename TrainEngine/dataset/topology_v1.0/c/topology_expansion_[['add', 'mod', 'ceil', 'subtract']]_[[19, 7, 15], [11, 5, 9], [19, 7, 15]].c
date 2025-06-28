void default_function_kernel(float* T_add, float* T_mod, float* T_subtract, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1995; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] + ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 19; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 7; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 15; ++ax2) {
        T_mod[(((ax0 * 105) + (ax1 * 15)) + ax2)] = fmodf(ph_0[(((ax0 * 105) + (ax1 * 15)) + ax2)], ph_3[(((ax0 * 105) + (ax1 * 15)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_1 = 0; ax0_1 < 19; ++ax0_1) {
    for (int32_t ax1_1 = 0; ax1_1 < 7; ++ax1_1) {
      for (int32_t ax2_1 = 0; ax2_1 < 15; ++ax2_1) {
        T_subtract[(((ax0_1 * 105) + (ax1_1 * 15)) + ax2_1)] = (ceilf(ph_0[(((ax0_1 * 105) + (ax1_1 * 15)) + ax2_1)]) - ph_0[(((ax0_1 * 105) + (ax1_1 * 15)) + ax2_1)]);
      }
    }
  }
}


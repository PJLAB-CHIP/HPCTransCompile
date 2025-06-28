void default_function_kernel(float* T_multiply, float* T_multiply_1, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 15; ++i0) {
    for (int32_t i1 = 0; i1 < 20; ++i1) {
      for (int32_t i2 = 0; i2 < 6; ++i2) {
        compute[(((i0 * 120) + (i1 * 6)) + i2)] = acosf(ph_0[(((i0 * 120) + (i1 * 6)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 15; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 20; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 6; ++ax2) {
        T_multiply[(((ax0 * 120) + (ax1 * 6)) + ax2)] = (ph_0[(((ax0 * 120) + (ax1 * 6)) + ax2)] * fmodf(fabsf(ph_0[(((ax0 * 120) + (ax1 * 6)) + ax2)]), ph_0[(((ax0 * 120) + (ax1 * 6)) + ax2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1800; ++ax0_ax1_fused_ax2_fused) {
    T_multiply_1[ax0_ax1_fused_ax2_fused] = (acoshf(ph_0[ax0_ax1_fused_ax2_fused]) * ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


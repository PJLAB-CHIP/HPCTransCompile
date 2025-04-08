void default_function_kernel(float* T_mod, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 18; ++i0) {
    for (int32_t i1 = 0; i1 < 9; ++i1) {
      for (int32_t i2 = 0; i2 < 14; ++i2) {
        compute[(((i0 * 126) + (i1 * 14)) + i2)] = asinhf(ph_0[(((i0 * 126) + (i1 * 14)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 2268; ++ax0_ax1_fused_ax2_fused) {
    T_mod[ax0_ax1_fused_ax2_fused] = fmodf(atanf(ph_0[ax0_ax1_fused_ax2_fused]), ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 18; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 9; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 14; ++i2_1) {
        compute_1[(((i0_1 * 126) + (i1_1 * 14)) + i2_1)] = expf(ph_0[(((i0_1 * 126) + (i1_1 * 14)) + i2_1)]);
      }
    }
  }
}


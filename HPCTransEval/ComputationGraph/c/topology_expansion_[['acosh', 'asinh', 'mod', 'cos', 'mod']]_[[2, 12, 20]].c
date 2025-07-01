void default_function_kernel(float* T_mod, float* T_mod_1, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 480; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = acoshf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 2; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 12; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 20; ++ax2) {
        T_mod[(((ax0 * 240) + (ax1 * 20)) + ax2)] = fmodf(asinhf(ph_0[(((ax0 * 240) + (ax1 * 20)) + ax2)]), ph_0[(((ax0 * 240) + (ax1 * 20)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_1 = 0; ax0_1 < 2; ++ax0_1) {
    for (int32_t ax1_1 = 0; ax1_1 < 12; ++ax1_1) {
      for (int32_t ax2_1 = 0; ax2_1 < 20; ++ax2_1) {
        T_mod_1[(((ax0_1 * 240) + (ax1_1 * 20)) + ax2_1)] = fmodf(ph_0[(((ax0_1 * 240) + (ax1_1 * 20)) + ax2_1)], cosf(ph_0[(((ax0_1 * 240) + (ax1_1 * 20)) + ax2_1)]));
      }
    }
  }
}


void default_function_kernel(float* T_mod, float* T_mod_1, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 10; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 19; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 14; ++ax2) {
        T_mod[(((ax0 * 266) + (ax1 * 14)) + ax2)] = fmodf(ph_0[(((ax0 * 266) + (ax1 * 14)) + ax2)], ph_3[(((ax0 * 266) + (ax1 * 14)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_1 = 0; ax0_1 < 10; ++ax0_1) {
    for (int32_t ax1_1 = 0; ax1_1 < 19; ++ax1_1) {
      for (int32_t ax2_1 = 0; ax2_1 < 14; ++ax2_1) {
        T_mod_1[(((ax0_1 * 266) + (ax1_1 * 14)) + ax2_1)] = fmodf((ph_0[(((ax0_1 * 266) + (ax1_1 * 14)) + ax2_1)] * (ph_0[(((ax0_1 * 266) + (ax1_1 * 14)) + ax2_1)] / ph_3[(((ax0_1 * 266) + (ax1_1 * 14)) + ax2_1)])), ph_0[(((ax0_1 * 266) + (ax1_1 * 14)) + ax2_1)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 2660; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf((ph_0[i0_i1_fused_i2_fused] * (ph_0[i0_i1_fused_i2_fused] / ph_3[i0_i1_fused_i2_fused])));
  }
}


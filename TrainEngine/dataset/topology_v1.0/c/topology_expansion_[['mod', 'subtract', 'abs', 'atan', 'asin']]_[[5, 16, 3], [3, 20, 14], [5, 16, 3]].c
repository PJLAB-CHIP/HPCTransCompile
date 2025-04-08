void default_function_kernel(float* T_mod, float* T_subtract, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 5; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 16; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 3; ++ax2) {
        T_mod[(((ax0 * 48) + (ax1 * 3)) + ax2)] = fmodf(ph_0[(((ax0 * 48) + (ax1 * 3)) + ax2)], ph_3[(((ax0 * 48) + (ax1 * 3)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 80; ++ax0_ax1_fused) {
    for (int32_t ax2_1 = 0; ax2_1 < 3; ++ax2_1) {
      T_subtract[((ax0_ax1_fused * 3) + ax2_1)] = (ph_0[((ax0_ax1_fused * 3) + ax2_1)] - ph_3[((ax0_ax1_fused * 3) + ax2_1)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 240; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = atanf(fabsf(ph_0[i0_i1_fused_i2_fused]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 240; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = asinf(fabsf(ph_0[i0_i1_fused_i2_fused_1]));
  }
}


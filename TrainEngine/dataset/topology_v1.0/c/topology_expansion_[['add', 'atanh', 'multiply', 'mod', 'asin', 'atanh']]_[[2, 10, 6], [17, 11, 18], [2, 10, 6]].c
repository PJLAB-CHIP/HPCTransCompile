void default_function_kernel(float* T_mod, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 2; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 6; ++ax2) {
        T_mod[(((ax0 * 60) + (ax1 * 6)) + ax2)] = fmodf((ph_0[(((ax0 * 60) + (ax1 * 6)) + ax2)] * atanhf(ph_0[(((ax0 * 60) + (ax1 * 6)) + ax2)])), ph_0[(((ax0 * 60) + (ax1 * 6)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 2; ++i0) {
    for (int32_t i1 = 0; i1 < 10; ++i1) {
      for (int32_t i2 = 0; i2 < 6; ++i2) {
        compute[(((i0 * 60) + (i1 * 6)) + i2)] = asinf((ph_0[(((i0 * 60) + (i1 * 6)) + i2)] * atanhf(ph_0[(((i0 * 60) + (i1 * 6)) + i2)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 120; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = atanhf((ph_0[i0_i1_fused_i2_fused] + ph_3[i0_i1_fused_i2_fused]));
  }
}


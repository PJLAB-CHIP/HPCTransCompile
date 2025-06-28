void default_function_kernel(float* T_add, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 140; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 6; ++i2) {
      compute[((i0_i1_fused * 6) + i2)] = sinf(ph_0[((i0_i1_fused * 6) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 7; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 20; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 6; ++ax2) {
        T_add[(((ax0 * 120) + (ax1 * 6)) + ax2)] = (ph_0[(((ax0 * 120) + (ax1 * 6)) + ax2)] + (asinhf(ph_0[(((ax0 * 120) + (ax1 * 6)) + ax2)]) * ph_0[(((ax0 * 120) + (ax1 * 6)) + ax2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 840; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = asinhf(atanf(ph_0[i0_i1_fused_i2_fused]));
  }
}


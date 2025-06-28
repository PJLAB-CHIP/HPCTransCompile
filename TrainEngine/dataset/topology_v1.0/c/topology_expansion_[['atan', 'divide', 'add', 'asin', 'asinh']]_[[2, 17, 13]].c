void default_function_kernel(float* T_add, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 2; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 17; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 13; ++ax2) {
        T_add[(((ax0 * 221) + (ax1 * 13)) + ax2)] = ((ph_0[(((ax0 * 221) + (ax1 * 13)) + ax2)] / atanf(ph_0[(((ax0 * 221) + (ax1 * 13)) + ax2)])) + ph_0[(((ax0 * 221) + (ax1 * 13)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 442; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 34; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 13; ++i2) {
      compute_1[((i0_i1_fused * 13) + i2)] = asinhf(ph_0[((i0_i1_fused * 13) + i2)]);
    }
  }
}


void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 60; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 6; ++i2) {
      compute[((i0_i1_fused * 6) + i2)] = acosf(ph_0[((i0_i1_fused * 6) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 360; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = asinf(atanhf(ph_0[i0_i1_fused_i2_fused]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 60; ++i0_i1_fused_1) {
    for (int32_t i2_1 = 0; i2_1 < 6; ++i2_1) {
      compute_2[((i0_i1_fused_1 * 6) + i2_1)] = atanf(ph_0[((i0_i1_fused_1 * 6) + i2_1)]);
    }
  }
}


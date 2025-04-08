void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 513; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinhf((ph_0[i0_i1_fused_i2_fused] / asinf(ph_0[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 513; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = ceilf(ph_0[i0_i1_fused_i2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 19; ++i0) {
    for (int32_t i1 = 0; i1 < 9; ++i1) {
      for (int32_t i2 = 0; i2 < 3; ++i2) {
        compute_2[(((i0 * 27) + (i1 * 3)) + i2)] = asinhf(asinhf(ph_0[(((i0 * 27) + (i1 * 3)) + i2)]));
      }
    }
  }
}


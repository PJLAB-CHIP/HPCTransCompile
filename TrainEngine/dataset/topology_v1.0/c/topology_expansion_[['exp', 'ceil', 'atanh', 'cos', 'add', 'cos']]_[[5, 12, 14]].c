void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 5; ++i0) {
    for (int32_t i1 = 0; i1 < 12; ++i1) {
      for (int32_t i2 = 0; i2 < 14; ++i2) {
        compute[(((i0 * 168) + (i1 * 14)) + i2)] = expf(ph_0[(((i0 * 168) + (i1 * 14)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 840; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = atanhf(ceilf(ph_0[i0_i1_fused_i2_fused]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 60; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 14; ++i2_1) {
      compute_2[((i0_i1_fused * 14) + i2_1)] = cosf((ph_0[((i0_i1_fused * 14) + i2_1)] + cosf(ph_0[((i0_i1_fused * 14) + i2_1)])));
    }
  }
}


void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* compute_3, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 4332; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = expf(fmodf(ph_0[i0_i1_fused_i2_fused], cosf(ph_0[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 4332; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = expf(fmodf(ph_0[i0_i1_fused_i2_fused_1], cosf(ph_0[i0_i1_fused_i2_fused_1])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 4332; ++i0_i1_fused_i2_fused_2) {
    compute_2[i0_i1_fused_i2_fused_2] = ceilf((ph_0[i0_i1_fused_i2_fused_2] + ph_3[i0_i1_fused_i2_fused_2]));
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 19; ++i0) {
    for (int32_t i1 = 0; i1 < 12; ++i1) {
      for (int32_t i2 = 0; i2 < 19; ++i2) {
        compute_3[(((i0 * 228) + (i1 * 19)) + i2)] = fabsf((ph_0[(((i0 * 228) + (i1 * 19)) + i2)] + ph_3[(((i0 * 228) + (i1 * 19)) + i2)]));
      }
    }
  }
}


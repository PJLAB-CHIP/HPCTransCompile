void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 14; ++i0) {
    for (int32_t i1 = 0; i1 < 11; ++i1) {
      for (int32_t i2 = 0; i2 < 4; ++i2) {
        compute[(((i0 * 44) + (i1 * 4)) + i2)] = asinhf((ph_0[(((i0 * 44) + (i1 * 4)) + i2)] * sinf(ph_0[(((i0 * 44) + (i1 * 4)) + i2)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 616; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = acosf((ph_0[i0_i1_fused_i2_fused] * sinf(ph_0[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 616; ++i0_i1_fused_i2_fused_1) {
    compute_2[i0_i1_fused_i2_fused_1] = atanf((ph_0[i0_i1_fused_i2_fused_1] - ph_3[i0_i1_fused_i2_fused_1]));
  }
}


void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* compute_3, float* compute_4, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 16; ++i0) {
    for (int32_t i1 = 0; i1 < 5; ++i1) {
      for (int32_t i2 = 0; i2 < 15; ++i2) {
        compute[(((i0 * 75) + (i1 * 15)) + i2)] = asinhf(ph_0[(((i0 * 75) + (i1 * 15)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1200; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = atanhf(ceilf(ph_0[i0_i1_fused_i2_fused]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 80; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 15; ++i2_1) {
      compute_2[((i0_i1_fused * 15) + i2_1)] = ceilf(ph_0[((i0_i1_fused * 15) + i2_1)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 16; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 5; ++i1_1) {
      for (int32_t i2_2 = 0; i2_2 < 15; ++i2_2) {
        compute_3[(((i0_1 * 75) + (i1_1 * 15)) + i2_2)] = asinhf(fabsf(ph_0[(((i0_1 * 75) + (i1_1 * 15)) + i2_2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 1200; ++i0_i1_fused_i2_fused_1) {
    compute_4[i0_i1_fused_i2_fused_1] = atanf(fabsf(ph_0[i0_i1_fused_i2_fused_1]));
  }
}


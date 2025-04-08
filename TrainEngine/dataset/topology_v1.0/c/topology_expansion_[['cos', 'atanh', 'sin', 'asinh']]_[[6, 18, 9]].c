void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 6; ++i0) {
    for (int32_t i1 = 0; i1 < 18; ++i1) {
      for (int32_t i2 = 0; i2 < 9; ++i2) {
        compute[(((i0 * 162) + (i1 * 9)) + i2)] = cosf(ph_0[(((i0 * 162) + (i1 * 9)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 972; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = sinf(atanhf(ph_0[i0_i1_fused_i2_fused]));
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 6; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 18; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 9; ++i2_1) {
        compute_2[(((i0_1 * 162) + (i1_1 * 9)) + i2_1)] = asinhf(ph_0[(((i0_1 * 162) + (i1_1 * 9)) + i2_1)]);
      }
    }
  }
}


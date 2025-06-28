void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 11; ++i0) {
    for (int32_t i1 = 0; i1 < 10; ++i1) {
      for (int32_t i2 = 0; i2 < 12; ++i2) {
        compute[(((i0 * 120) + (i1 * 12)) + i2)] = fabsf((ph_0[(((i0 * 120) + (i1 * 12)) + i2)] - fabsf(ph_0[(((i0 * 120) + (i1 * 12)) + i2)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 11; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 10; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 12; ++i2_1) {
        compute_1[(((i0_1 * 120) + (i1_1 * 12)) + i2_1)] = atanf(ph_0[(((i0_1 * 120) + (i1_1 * 12)) + i2_1)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1320; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = ceilf(ceilf(ph_0[i0_i1_fused_i2_fused]));
  }
}


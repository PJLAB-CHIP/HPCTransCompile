void default_function_kernel(float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 9; ++i0) {
    for (int32_t i1 = 0; i1 < 18; ++i1) {
      for (int32_t i2 = 0; i2 < 11; ++i2) {
        compute[(((i0 * 198) + (i1 * 11)) + i2)] = asinf(ph_0[(((i0 * 198) + (i1 * 11)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 162; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 11; ++i2_1) {
      compute_1[((i0_i1_fused * 11) + i2_1)] = asinhf(ph_0[((i0_i1_fused * 11) + i2_1)]);
    }
  }
}


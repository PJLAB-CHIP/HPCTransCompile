void default_function_kernel(float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 3; ++i0) {
    for (int32_t i1 = 0; i1 < 12; ++i1) {
      for (int32_t i2 = 0; i2 < 13; ++i2) {
        compute[(((i0 * 156) + (i1 * 13)) + i2)] = acosf(ph_0[(((i0 * 156) + (i1 * 13)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 3; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 12; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 13; ++i2_1) {
        compute_1[(((i0_1 * 156) + (i1_1 * 13)) + i2_1)] = fabsf(ph_0[(((i0_1 * 156) + (i1_1 * 13)) + i2_1)]);
      }
    }
  }
}


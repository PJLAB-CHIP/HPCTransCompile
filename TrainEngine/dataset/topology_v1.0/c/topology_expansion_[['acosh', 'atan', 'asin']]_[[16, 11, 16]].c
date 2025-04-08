void default_function_kernel(float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 16; ++i0) {
    for (int32_t i1 = 0; i1 < 11; ++i1) {
      for (int32_t i2 = 0; i2 < 16; ++i2) {
        compute[(((i0 * 176) + (i1 * 16)) + i2)] = acoshf(ph_0[(((i0 * 176) + (i1 * 16)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 16; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 11; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 16; ++i2_1) {
        compute_1[(((i0_1 * 176) + (i1_1 * 16)) + i2_1)] = asinf(atanf(ph_0[(((i0_1 * 176) + (i1_1 * 16)) + i2_1)]));
      }
    }
  }
}


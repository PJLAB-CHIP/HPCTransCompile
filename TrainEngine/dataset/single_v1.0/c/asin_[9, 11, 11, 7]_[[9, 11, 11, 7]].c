void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 9; ++i0) {
    for (int32_t i1 = 0; i1 < 11; ++i1) {
      for (int32_t i2 = 0; i2 < 11; ++i2) {
        for (int32_t i3_s = 0; i3_s < 7; ++i3_s) {
          compute[((((i0 * 847) + (i1 * 77)) + (i2 * 7)) + i3_s)] = asinf(data[((((i0 * 847) + (i1 * 77)) + (i2 * 7)) + i3_s)]);
        }
      }
    }
  }
}


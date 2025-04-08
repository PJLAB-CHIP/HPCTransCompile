void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 11; ++i0) {
    for (int32_t i1 = 0; i1 < 6; ++i1) {
      for (int32_t i2 = 0; i2 < 2; ++i2) {
        for (int32_t i3 = 0; i3 < 18; ++i3) {
          compute[((((i0 * 216) + (i1 * 36)) + (i2 * 18)) + i3)] = tanf(data[((((i0 * 216) + (i1 * 36)) + (i2 * 18)) + i3)]);
        }
      }
    }
  }
}


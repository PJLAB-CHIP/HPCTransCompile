void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 2; ++i0) {
    for (int32_t i1 = 0; i1 < 15; ++i1) {
      for (int32_t i2 = 0; i2 < 2; ++i2) {
        for (int32_t i3 = 0; i3 < 8; ++i3) {
          compute[((((i0 * 240) + (i1 * 16)) + (i2 * 8)) + i3)] = coshf(data[((((i0 * 240) + (i1 * 16)) + (i2 * 8)) + i3)]);
        }
      }
    }
  }
}


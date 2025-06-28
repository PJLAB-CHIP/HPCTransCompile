void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 15; ++i0) {
    for (int32_t i1 = 0; i1 < 2; ++i1) {
      for (int32_t i2 = 0; i2 < 8; ++i2) {
        for (int32_t i3 = 0; i3 < 20; ++i3) {
          compute[((((i0 * 320) + (i1 * 160)) + (i2 * 20)) + i3)] = tanf(data[((((i0 * 320) + (i1 * 160)) + (i2 * 20)) + i3)]);
        }
      }
    }
  }
}


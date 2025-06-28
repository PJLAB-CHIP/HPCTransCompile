void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 4; ++i0) {
    for (int32_t i1 = 0; i1 < 10; ++i1) {
      for (int32_t i2 = 0; i2 < 20; ++i2) {
        for (int32_t i3 = 0; i3 < 4; ++i3) {
          compute[((((i0 * 800) + (i1 * 80)) + (i2 * 4)) + i3)] = log10f(data[((((i0 * 800) + (i1 * 80)) + (i2 * 4)) + i3)]);
        }
      }
    }
  }
}


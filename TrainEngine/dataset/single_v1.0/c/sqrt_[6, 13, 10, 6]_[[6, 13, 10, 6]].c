void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 6; ++i0) {
    for (int32_t i1 = 0; i1 < 13; ++i1) {
      for (int32_t i2 = 0; i2 < 10; ++i2) {
        for (int32_t i3 = 0; i3 < 6; ++i3) {
          compute[((((i0 * 780) + (i1 * 60)) + (i2 * 6)) + i3)] = sqrtf(data[((((i0 * 780) + (i1 * 60)) + (i2 * 6)) + i3)]);
        }
      }
    }
  }
}


void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 13; ++i0) {
    for (int32_t i1 = 0; i1 < 13; ++i1) {
      for (int32_t i2 = 0; i2 < 3; ++i2) {
        for (int32_t i3 = 0; i3 < 15; ++i3) {
          compute[((((i0 * 585) + (i1 * 45)) + (i2 * 15)) + i3)] = ceilf(data[((((i0 * 585) + (i1 * 45)) + (i2 * 15)) + i3)]);
        }
      }
    }
  }
}


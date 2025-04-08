void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 3; ++i0) {
    for (int32_t i1 = 0; i1 < 16; ++i1) {
      for (int32_t i2 = 0; i2 < 9; ++i2) {
        for (int32_t i3 = 0; i3 < 15; ++i3) {
          compute[((((i0 * 2160) + (i1 * 135)) + (i2 * 15)) + i3)] = erff(data[((((i0 * 2160) + (i1 * 135)) + (i2 * 15)) + i3)]);
        }
      }
    }
  }
}


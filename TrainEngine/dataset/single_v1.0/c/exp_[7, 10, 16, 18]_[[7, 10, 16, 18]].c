void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 7; ++i0) {
    for (int32_t i1 = 0; i1 < 10; ++i1) {
      for (int32_t i2 = 0; i2 < 16; ++i2) {
        for (int32_t i3 = 0; i3 < 18; ++i3) {
          compute[((((i0 * 2880) + (i1 * 288)) + (i2 * 18)) + i3)] = expf(data[((((i0 * 2880) + (i1 * 288)) + (i2 * 18)) + i3)]);
        }
      }
    }
  }
}


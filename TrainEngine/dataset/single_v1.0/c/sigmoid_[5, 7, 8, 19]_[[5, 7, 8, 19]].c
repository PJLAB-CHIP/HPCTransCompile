void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 5; ++i0) {
    for (int32_t i1 = 0; i1 < 7; ++i1) {
      for (int32_t i2 = 0; i2 < 8; ++i2) {
        for (int32_t i3 = 0; i3 < 19; ++i3) {
          compute[((((i0 * 1064) + (i1 * 152)) + (i2 * 19)) + i3)] = (1.000000e+00f / (1.000000e+00f + expf((0.000000e+00f - data[((((i0 * 1064) + (i1 * 152)) + (i2 * 19)) + i3)]))));
        }
      }
    }
  }
}


void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 13; ++i0) {
    for (int32_t i1 = 0; i1 < 6; ++i1) {
      for (int32_t i2 = 0; i2 < 14; ++i2) {
        for (int32_t i3_s = 0; i3_s < 15; ++i3_s) {
          compute[((((i0 * 1260) + (i1 * 210)) + (i2 * 15)) + i3_s)] = asinhf(data[((((i0 * 1260) + (i1 * 210)) + (i2 * 15)) + i3_s)]);
        }
      }
    }
  }
}


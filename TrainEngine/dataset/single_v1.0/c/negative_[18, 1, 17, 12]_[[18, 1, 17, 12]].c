void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 18; ++i0) {
    for (int32_t i2 = 0; i2 < 17; ++i2) {
      for (int32_t i3 = 0; i3 < 12; ++i3) {
        compute[(((i0 * 204) + (i2 * 12)) + i3)] = (data[(((i0 * 204) + (i2 * 12)) + i3)] * -1.000000e+00f);
      }
    }
  }
}


void default_function_kernel(float* compute, float* data) {
  for (int32_t i1 = 0; i1 < 10; ++i1) {
    for (int32_t i2 = 0; i2 < 9; ++i2) {
      for (int32_t i3 = 0; i3 < 14; ++i3) {
        compute[(((i1 * 126) + (i2 * 14)) + i3)] = sinf(data[(((i1 * 126) + (i2 * 14)) + i3)]);
      }
    }
  }
}


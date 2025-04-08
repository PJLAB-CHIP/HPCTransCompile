void default_function_kernel(float* compute, float* data) {
  for (int32_t i1 = 0; i1 < 18; ++i1) {
    for (int32_t i2 = 0; i2 < 14; ++i2) {
      for (int32_t i3 = 0; i3 < 7; ++i3) {
        compute[(((i1 * 98) + (i2 * 7)) + i3)] = atanf(data[(((i1 * 98) + (i2 * 7)) + i3)]);
      }
    }
  }
}


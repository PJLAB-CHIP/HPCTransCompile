void default_function_kernel(float* compute, float* data) {
  for (int32_t i1 = 0; i1 < 15; ++i1) {
    for (int32_t i2 = 0; i2 < 14; ++i2) {
      for (int32_t i3 = 0; i3 < 6; ++i3) {
        compute[(((i1 * 84) + (i2 * 6)) + i3)] = tanf(data[(((i1 * 84) + (i2 * 6)) + i3)]);
      }
    }
  }
}


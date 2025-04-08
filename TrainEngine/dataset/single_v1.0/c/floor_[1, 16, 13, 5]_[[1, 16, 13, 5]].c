void default_function_kernel(float* compute, float* data) {
  for (int32_t i1 = 0; i1 < 16; ++i1) {
    for (int32_t i2 = 0; i2 < 13; ++i2) {
      for (int32_t i3 = 0; i3 < 5; ++i3) {
        compute[(((i1 * 65) + (i2 * 5)) + i3)] = floorf(data[(((i1 * 65) + (i2 * 5)) + i3)]);
      }
    }
  }
}


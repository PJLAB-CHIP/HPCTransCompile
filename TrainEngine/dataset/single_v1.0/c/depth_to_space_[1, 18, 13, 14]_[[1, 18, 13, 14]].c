void default_function_kernel(float* data, float* depth_to_space) {
  for (int32_t i1 = 0; i1 < 4; ++i1) {
    for (int32_t i2 = 0; i2 < 26; ++i2) {
      for (int32_t i3 = 0; i3 < 28; ++i3) {
        depth_to_space[(((i1 * 728) + (i2 * 28)) + i3)] = data[((((((i2 % 2) * 1456) + ((i3 % 2) * 728)) + (i1 * 182)) + ((i2 / 2) * 14)) + (i3 / 2))];
      }
    }
  }
}


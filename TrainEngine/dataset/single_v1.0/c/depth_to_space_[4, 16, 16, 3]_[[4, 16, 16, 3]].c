void default_function_kernel(float* data, float* depth_to_space) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 4; ++i0) {
    for (int32_t i1 = 0; i1 < 4; ++i1) {
      for (int32_t i2 = 0; i2 < 32; ++i2) {
        for (int32_t i3 = 0; i3 < 6; ++i3) {
          depth_to_space[((((i0 * 768) + (i1 * 192)) + (i2 * 6)) + i3)] = data[((((((i0 * 768) + ((i2 % 2) * 384)) + ((i3 % 2) * 192)) + (i1 * 48)) + ((i2 / 2) * 3)) + (i3 / 2))];
        }
      }
    }
  }
}


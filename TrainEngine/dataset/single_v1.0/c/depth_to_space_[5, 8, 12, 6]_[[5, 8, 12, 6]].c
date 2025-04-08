void default_function_kernel(float* data, float* depth_to_space) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 5; ++i0) {
    for (int32_t i1 = 0; i1 < 2; ++i1) {
      for (int32_t i2 = 0; i2 < 24; ++i2) {
        for (int32_t i3 = 0; i3 < 12; ++i3) {
          depth_to_space[((((i0 * 576) + (i1 * 288)) + (i2 * 12)) + i3)] = data[((((((i0 * 576) + ((i2 % 2) * 288)) + ((i3 % 2) * 144)) + (i1 * 72)) + ((i2 / 2) * 6)) + (i3 / 2))];
        }
      }
    }
  }
}


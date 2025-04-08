void default_function_kernel(float* compute, float* data) {
  for (int32_t i1 = 0; i1 < 16; ++i1) {
    for (int32_t i2 = 0; i2 < 13; ++i2) {
      compute[((i1 * 13) + i2)] = roundf(data[((i1 * 13) + i2)]);
    }
  }
}


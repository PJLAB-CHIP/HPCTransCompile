void default_function_kernel(float* compute, float* compute_1, float* data, float* data_1) {
  for (int32_t i1 = 0; i1 < 14; ++i1) {
    for (int32_t i2 = 0; i2 < 6; ++i2) {
      for (int32_t i3 = 0; i3 < 10; ++i3) {
        compute[(((i1 * 60) + (i2 * 10)) + i3)] = sqrtf((data[(((i1 * 60) + (i2 * 10)) + i3)] + data_1[(((i1 * 60) + (i2 * 10)) + i3)]));
      }
    }
  }
  for (int32_t i1_1 = 0; i1_1 < 14; ++i1_1) {
    for (int32_t i2_1 = 0; i2_1 < 6; ++i2_1) {
      for (int32_t i3_1 = 0; i3_1 < 10; ++i3_1) {
        compute_1[(((i1_1 * 60) + (i2_1 * 10)) + i3_1)] = cosf((data[(((i1_1 * 60) + (i2_1 * 10)) + i3_1)] + data_1[(((i1_1 * 60) + (i2_1 * 10)) + i3_1)]));
      }
    }
  }
}


void default_function_kernel(float* compute, float* compute_1, float* data, float* data_1) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 4; ++i0) {
    for (int32_t i1 = 0; i1 < 17; ++i1) {
      for (int32_t i2 = 0; i2 < 12; ++i2) {
        for (int32_t i3 = 0; i3 < 17; ++i3) {
          compute[((((i0 * 3468) + (i1 * 204)) + (i2 * 17)) + i3)] = sqrtf((data[((((i0 * 3468) + (i1 * 204)) + (i2 * 17)) + i3)] + data_1[((((i0 * 3468) + (i1 * 204)) + (i2 * 17)) + i3)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 68; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 12; ++i2_1) {
      for (int32_t i3_1 = 0; i3_1 < 17; ++i3_1) {
        compute_1[(((i0_i1_fused * 204) + (i2_1 * 17)) + i3_1)] = cosf((data[(((i0_i1_fused * 204) + (i2_1 * 17)) + i3_1)] + data_1[(((i0_i1_fused * 204) + (i2_1 * 17)) + i3_1)]));
      }
    }
  }
}


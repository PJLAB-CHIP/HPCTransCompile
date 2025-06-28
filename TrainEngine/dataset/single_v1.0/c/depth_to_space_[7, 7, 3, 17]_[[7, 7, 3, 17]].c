void default_function_kernel(float* data, float* depth_to_space) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 7; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 6; ++i2) {
      for (int32_t i3 = 0; i3 < 34; ++i3) {
        depth_to_space[(((i0_i1_fused * 204) + (i2 * 34)) + i3)] = data[(((((i0_i1_fused * 357) + ((i2 % 2) * 102)) + ((i3 % 2) * 51)) + ((i2 / 2) * 17)) + (i3 / 2))];
      }
    }
  }
}


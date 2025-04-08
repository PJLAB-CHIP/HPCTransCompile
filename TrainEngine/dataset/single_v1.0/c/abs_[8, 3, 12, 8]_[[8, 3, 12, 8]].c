void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 24; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 12; ++i2) {
      for (int32_t i3 = 0; i3 < 8; ++i3) {
        compute[(((i0_i1_fused * 96) + (i2 * 8)) + i3)] = fabsf(data[(((i0_i1_fused * 96) + (i2 * 8)) + i3)]);
      }
    }
  }
}


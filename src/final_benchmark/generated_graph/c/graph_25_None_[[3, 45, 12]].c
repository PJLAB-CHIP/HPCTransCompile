void default_function_kernel(float* compute, float* ph) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 45; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 12; ++i2) {
      for (int32_t i3_inner = 0; i3_inner < 3; ++i3_inner) {
        compute[(((i0_i1_fused * 36) + (i2 * 3)) + i3_inner)] = max(min(floorf(ph[(((i3_inner * 540) + (i0_i1_fused * 12)) + i2)]), 1.111000e+01f), 1.100000e+00f);
      }
    }
  }
}


void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 75; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 5; ++i2) {
      for (int32_t i3 = 0; i3 < 12; ++i3) {
        compute[(((i0_i1_fused * 60) + (i2 * 12)) + i3)] = coshf(data[(((i0_i1_fused * 60) + (i2 * 12)) + i3)]);
      }
    }
  }
}


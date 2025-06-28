void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 32; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 15; ++i2) {
      for (int32_t i3 = 0; i3 < 7; ++i3) {
        compute[(((i0_i1_fused * 105) + (i2 * 7)) + i3)] = ((0.000000e+00f < data[(((i0_i1_fused * 105) + (i2 * 7)) + i3)]) ? data[(((i0_i1_fused * 105) + (i2 * 7)) + i3)] : (data[(((i0_i1_fused * 105) + (i2 * 7)) + i3)] * 5.000000e-01f));
      }
    }
  }
}


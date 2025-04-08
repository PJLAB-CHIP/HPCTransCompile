void default_function_kernel(float* data, float* resize) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 2508; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 24; ++i3) {
      resize[((i0_i1_fused_i2_fused * 24) + i3)] = data[((((i0_i1_fused_i2_fused / 12) * 72) + (((i0_i1_fused_i2_fused % 12) / 2) * 12)) + (i3 / 2))];
    }
  }
}


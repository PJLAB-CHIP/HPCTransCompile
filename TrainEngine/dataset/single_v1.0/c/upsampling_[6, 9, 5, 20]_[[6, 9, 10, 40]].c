void default_function_kernel(float* data, float* resize) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1080; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 80; ++i3) {
      resize[((i0_i1_fused_i2_fused * 80) + i3)] = data[((((i0_i1_fused_i2_fused / 20) * 400) + (((i0_i1_fused_i2_fused % 20) / 2) * 40)) + (i3 / 2))];
    }
  }
}


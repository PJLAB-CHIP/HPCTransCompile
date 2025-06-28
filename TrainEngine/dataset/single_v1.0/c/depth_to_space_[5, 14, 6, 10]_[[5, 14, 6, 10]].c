void default_function_kernel(float* data, float* depth_to_space) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 180; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 20; ++i3) {
      depth_to_space[((i0_i1_fused_i2_fused * 20) + i3)] = data[(((((((i0_i1_fused_i2_fused / 36) * 840) + (((i0_i1_fused_i2_fused % 12) % 2) * 360)) + ((i3 % 2) * 180)) + (((i0_i1_fused_i2_fused % 36) / 12) * 60)) + (((i0_i1_fused_i2_fused % 12) / 2) * 10)) + (i3 / 2))];
    }
  }
}


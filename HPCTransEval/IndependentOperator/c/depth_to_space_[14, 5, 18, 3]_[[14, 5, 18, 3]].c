void default_function_kernel(float* data, float* depth_to_space) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 504; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 6; ++i3) {
      depth_to_space[((i0_i1_fused_i2_fused * 6) + i3)] = data[((((((i0_i1_fused_i2_fused / 36) * 270) + (((i0_i1_fused_i2_fused % 36) % 2) * 108)) + ((i3 % 2) * 54)) + (((i0_i1_fused_i2_fused % 36) / 2) * 3)) + (i3 / 2))];
    }
  }
}


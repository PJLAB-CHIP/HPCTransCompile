void default_function_kernel(float* data, float* depth_to_space) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 210; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 8; ++i3) {
      depth_to_space[((i0_i1_fused_i2_fused * 8) + i3)] = data[(((((((i0_i1_fused_i2_fused / 42) * 364) + (((i0_i1_fused_i2_fused % 14) % 2) * 168)) + ((i3 % 2) * 84)) + (((i0_i1_fused_i2_fused % 42) / 14) * 28)) + (((i0_i1_fused_i2_fused % 14) / 2) * 4)) + (i3 / 2))];
    }
  }
}


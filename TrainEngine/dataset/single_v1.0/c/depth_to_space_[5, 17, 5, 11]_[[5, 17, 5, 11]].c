void default_function_kernel(float* data, float* depth_to_space) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 200; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 22; ++i3) {
      depth_to_space[((i0_i1_fused_i2_fused * 22) + i3)] = data[(((((((i0_i1_fused_i2_fused / 40) * 935) + (((i0_i1_fused_i2_fused % 10) % 2) * 440)) + ((i3 % 2) * 220)) + (((i0_i1_fused_i2_fused % 40) / 10) * 55)) + (((i0_i1_fused_i2_fused % 10) / 2) * 11)) + (i3 / 2))];
    }
  }
}


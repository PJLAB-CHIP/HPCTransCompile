void default_function_kernel(float* DilatedInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 360; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 4; ++i3) {
      DilatedInput[((i0_i1_fused_i2_fused * 4) + i3)] = data[((i0_i1_fused_i2_fused * 4) + i3)];
    }
  }
}


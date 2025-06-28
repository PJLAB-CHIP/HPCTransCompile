void default_function_kernel(float* compute, float* compute_1, float* data, float* data_1) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 11; ++i0) {
    for (int32_t i1 = 0; i1 < 8; ++i1) {
      for (int32_t i2 = 0; i2 < 18; ++i2) {
        for (int32_t i3 = 0; i3 < 17; ++i3) {
          compute[((((i0 * 2448) + (i1 * 306)) + (i2 * 17)) + i3)] = sqrtf((data[((((i0 * 2448) + (i1 * 306)) + (i2 * 17)) + i3)] + data_1[((((i0 * 2448) + (i1 * 306)) + (i2 * 17)) + i3)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_i3_fused = 0; i0_i1_fused_i2_fused_i3_fused < 26928; ++i0_i1_fused_i2_fused_i3_fused) {
    compute_1[i0_i1_fused_i2_fused_i3_fused] = cosf((data[i0_i1_fused_i2_fused_i3_fused] + data_1[i0_i1_fused_i2_fused_i3_fused]));
  }
}


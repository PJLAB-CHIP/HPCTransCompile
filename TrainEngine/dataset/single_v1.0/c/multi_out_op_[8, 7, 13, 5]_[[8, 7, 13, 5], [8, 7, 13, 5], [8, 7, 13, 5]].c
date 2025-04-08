void default_function_kernel(float* compute, float* compute_1, float* data, float* data_1) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 56; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 13; ++i2) {
      for (int32_t i3 = 0; i3 < 5; ++i3) {
        compute[(((i0_i1_fused * 65) + (i2 * 5)) + i3)] = sqrtf((data[(((i0_i1_fused * 65) + (i2 * 5)) + i3)] + data_1[(((i0_i1_fused * 65) + (i2 * 5)) + i3)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_i3_fused = 0; i0_i1_fused_i2_fused_i3_fused < 3640; ++i0_i1_fused_i2_fused_i3_fused) {
    compute_1[i0_i1_fused_i2_fused_i3_fused] = cosf((data[i0_i1_fused_i2_fused_i3_fused] + data_1[i0_i1_fused_i2_fused_i3_fused]));
  }
}


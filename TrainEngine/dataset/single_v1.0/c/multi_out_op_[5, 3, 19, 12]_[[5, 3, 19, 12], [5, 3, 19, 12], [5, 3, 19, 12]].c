void default_function_kernel(float* compute, float* compute_1, float* data, float* data_1) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_i3_fused = 0; i0_i1_fused_i2_fused_i3_fused < 3420; ++i0_i1_fused_i2_fused_i3_fused) {
    compute[i0_i1_fused_i2_fused_i3_fused] = sqrtf((data[i0_i1_fused_i2_fused_i3_fused] + data_1[i0_i1_fused_i2_fused_i3_fused]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_i3_fused_1 = 0; i0_i1_fused_i2_fused_i3_fused_1 < 3420; ++i0_i1_fused_i2_fused_i3_fused_1) {
    compute_1[i0_i1_fused_i2_fused_i3_fused_1] = cosf((data[i0_i1_fused_i2_fused_i3_fused_1] + data_1[i0_i1_fused_i2_fused_i3_fused_1]));
  }
}


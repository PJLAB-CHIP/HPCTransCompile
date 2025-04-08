void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1188; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = sinf((ph_0[i0_i1_fused_i2_fused] + (ph_0[i0_i1_fused_i2_fused] - ph_3[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 1188; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = sinf((ph_0[i0_i1_fused_i2_fused_1] + (ph_0[i0_i1_fused_i2_fused_1] - ph_3[i0_i1_fused_i2_fused_1])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 132; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 9; ++i2) {
      compute_2[((i0_i1_fused * 9) + i2)] = acoshf((ph_0[((i0_i1_fused * 9) + i2)] / ph_3[((i0_i1_fused * 9) + i2)]));
    }
  }
}


void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 117; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 14; ++i2) {
      compute[((i0_i1_fused * 14) + i2)] = fabsf((ph_0[((i0_i1_fused * 14) + i2)] - sinf(ph_0[((i0_i1_fused * 14) + i2)])));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 117; ++i0_i1_fused_1) {
    for (int32_t i2_1 = 0; i2_1 < 14; ++i2_1) {
      compute_1[((i0_i1_fused_1 * 14) + i2_1)] = ceilf(ph_0[((i0_i1_fused_1 * 14) + i2_1)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1638; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = cosf(sinf(ph_0[i0_i1_fused_i2_fused]));
  }
}


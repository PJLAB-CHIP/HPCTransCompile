void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* compute_3, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 2016; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = atanf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 144; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 14; ++i2) {
      compute_1[((i0_i1_fused * 14) + i2)] = cosf(acosf(ph_0[((i0_i1_fused * 14) + i2)]));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 2016; ++i0_i1_fused_i2_fused_1) {
    compute_2[i0_i1_fused_i2_fused_1] = ceilf((ph_0[i0_i1_fused_i2_fused_1] * atanf(ph_0[i0_i1_fused_i2_fused_1])));
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 8; ++i0) {
    for (int32_t i1 = 0; i1 < 18; ++i1) {
      for (int32_t i2_1 = 0; i2_1 < 14; ++i2_1) {
        compute_3[(((i0 * 252) + (i1 * 14)) + i2_1)] = ceilf((ph_0[(((i0 * 252) + (i1 * 14)) + i2_1)] * atanf(ph_0[(((i0 * 252) + (i1 * 14)) + i2_1)])));
      }
    }
  }
}


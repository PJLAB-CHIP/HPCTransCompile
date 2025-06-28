void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* compute_3, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1001; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = ceilf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 11; ++i0) {
    for (int32_t i1 = 0; i1 < 13; ++i1) {
      for (int32_t i2 = 0; i2 < 7; ++i2) {
        compute_1[(((i0 * 91) + (i1 * 7)) + i2)] = fabsf(asinf(ph_0[(((i0 * 91) + (i1 * 7)) + i2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 143; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 7; ++i2_1) {
      compute_2[((i0_i1_fused * 7) + i2_1)] = asinf(ph_0[((i0_i1_fused * 7) + i2_1)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 1001; ++i0_i1_fused_i2_fused_1) {
    compute_3[i0_i1_fused_i2_fused_1] = acoshf(ceilf(ph_0[i0_i1_fused_i2_fused_1]));
  }
}


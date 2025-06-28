void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* compute_3, float* ph_0) {
  for (int32_t i1 = 0; i1 < 7; ++i1) {
    for (int32_t i2 = 0; i2 < 7; ++i2) {
      compute[((i1 * 7) + i2)] = sinf(ph_0[((i1 * 7) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 7; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 7; ++i2_1) {
      compute_1[((i0_i1_fused * 7) + i2_1)] = acoshf(ceilf(ph_0[((i0_i1_fused * 7) + i2_1)]));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 49; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 49; ++i0_i1_fused_i2_fused_1) {
    compute_3[i0_i1_fused_i2_fused_1] = acosf(atanhf(ph_0[i0_i1_fused_i2_fused_1]));
  }
}


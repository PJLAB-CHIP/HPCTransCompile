void default_function_kernel(float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 221; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 7; ++i2) {
      compute[((i0_i1_fused * 7) + i2)] = expf(ph_0[((i0_i1_fused * 7) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 221; ++i0_i1_fused_1) {
    for (int32_t i2_1 = 0; i2_1 < 7; ++i2_1) {
      compute_1[((i0_i1_fused_1 * 7) + i2_1)] = atanhf(ph_0[((i0_i1_fused_1 * 7) + i2_1)]);
    }
  }
}


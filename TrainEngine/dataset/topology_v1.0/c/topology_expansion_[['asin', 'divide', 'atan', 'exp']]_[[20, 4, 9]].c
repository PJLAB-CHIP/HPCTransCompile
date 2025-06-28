void default_function_kernel(float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 80; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 9; ++i2) {
      compute[((i0_i1_fused * 9) + i2)] = atanf((ph_0[((i0_i1_fused * 9) + i2)] / asinf(ph_0[((i0_i1_fused * 9) + i2)])));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 80; ++i0_i1_fused_1) {
    for (int32_t i2_1 = 0; i2_1 < 9; ++i2_1) {
      compute_1[((i0_i1_fused_1 * 9) + i2_1)] = expf(ph_0[((i0_i1_fused_1 * 9) + i2_1)]);
    }
  }
}


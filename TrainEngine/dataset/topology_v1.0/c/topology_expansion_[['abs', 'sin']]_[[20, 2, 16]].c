void default_function_kernel(float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 40; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 16; ++i2) {
      compute[((i0_i1_fused * 16) + i2)] = fabsf(ph_0[((i0_i1_fused * 16) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 40; ++i0_i1_fused_1) {
    for (int32_t i2_1 = 0; i2_1 < 16; ++i2_1) {
      compute_1[((i0_i1_fused_1 * 16) + i2_1)] = sinf(ph_0[((i0_i1_fused_1 * 16) + i2_1)]);
    }
  }
}


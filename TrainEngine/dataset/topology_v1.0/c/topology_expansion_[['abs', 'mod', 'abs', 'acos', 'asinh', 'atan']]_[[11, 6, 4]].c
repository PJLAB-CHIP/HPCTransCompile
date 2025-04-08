void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 11; ++i0) {
    for (int32_t i1 = 0; i1 < 6; ++i1) {
      for (int32_t i2 = 0; i2 < 4; ++i2) {
        compute[(((i0 * 24) + (i1 * 4)) + i2)] = fabsf(fmodf(ph_0[(((i0 * 24) + (i1 * 4)) + i2)], fabsf(ph_0[(((i0 * 24) + (i1 * 4)) + i2)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 264; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = acosf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 66; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 4; ++i2_1) {
      compute_2[((i0_i1_fused * 4) + i2_1)] = atanf(asinhf(ph_0[((i0_i1_fused * 4) + i2_1)]));
    }
  }
}


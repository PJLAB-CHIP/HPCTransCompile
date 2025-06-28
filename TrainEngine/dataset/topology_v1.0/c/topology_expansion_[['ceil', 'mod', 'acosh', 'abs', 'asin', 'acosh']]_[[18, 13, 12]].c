void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 18; ++i0) {
    for (int32_t i1 = 0; i1 < 13; ++i1) {
      for (int32_t i2 = 0; i2 < 12; ++i2) {
        compute[(((i0 * 156) + (i1 * 12)) + i2)] = acoshf(fmodf(ph_0[(((i0 * 156) + (i1 * 12)) + i2)], ceilf(ph_0[(((i0 * 156) + (i1 * 12)) + i2)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 2808; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = fabsf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 234; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 12; ++i2_1) {
      compute_2[((i0_i1_fused * 12) + i2_1)] = acoshf(asinf(ph_0[((i0_i1_fused * 12) + i2_1)]));
    }
  }
}


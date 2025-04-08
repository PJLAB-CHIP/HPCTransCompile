void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 16; ++i0) {
    for (int32_t i1 = 0; i1 < 14; ++i1) {
      for (int32_t i2 = 0; i2 < 2; ++i2) {
        compute[(((i0 * 28) + (i1 * 2)) + i2)] = ceilf(fmodf(ph_0[(((i0 * 28) + (i1 * 2)) + i2)], fabsf(ph_0[(((i0 * 28) + (i1 * 2)) + i2)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 16; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 14; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 2; ++i2_1) {
        compute_1[(((i0_1 * 28) + (i1_1 * 2)) + i2_1)] = atanf(ph_0[(((i0_1 * 28) + (i1_1 * 2)) + i2_1)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 448; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = atanf(ph_0[i0_i1_fused_i2_fused]);
  }
}


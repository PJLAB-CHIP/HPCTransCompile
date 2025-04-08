void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* compute_3, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 2640; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = acosf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 15; ++i0) {
    for (int32_t i1 = 0; i1 < 11; ++i1) {
      for (int32_t i2 = 0; i2 < 16; ++i2) {
        compute_1[(((i0 * 176) + (i1 * 16)) + i2)] = acosf(atanhf(ph_0[(((i0 * 176) + (i1 * 16)) + i2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 15; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 11; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 16; ++i2_1) {
        compute_2[(((i0_1 * 176) + (i1_1 * 16)) + i2_1)] = acosf(ph_0[(((i0_1 * 176) + (i1_1 * 16)) + i2_1)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 2640; ++i0_i1_fused_i2_fused_1) {
    compute_3[i0_i1_fused_i2_fused_1] = fabsf(ph_0[i0_i1_fused_i2_fused_1]);
  }
}


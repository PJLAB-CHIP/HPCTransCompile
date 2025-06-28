void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 17; ++i0) {
    for (int32_t i1 = 0; i1 < 13; ++i1) {
      for (int32_t i2 = 0; i2 < 18; ++i2) {
        compute[(((i0 * 234) + (i1 * 18)) + i2)] = acoshf((ph_0[(((i0 * 234) + (i1 * 18)) + i2)] + ceilf(ph_0[(((i0 * 234) + (i1 * 18)) + i2)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 17; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 13; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 18; ++i2_1) {
        compute_1[(((i0_1 * 234) + (i1_1 * 18)) + i2_1)] = ceilf(ph_0[(((i0_1 * 234) + (i1_1 * 18)) + i2_1)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 3978; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = asinhf(cosf(ph_0[i0_i1_fused_i2_fused]));
  }
}


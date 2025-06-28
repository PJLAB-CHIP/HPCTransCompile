void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* compute_3, float* compute_4, float* ph_0, float* ph_3) {
  float compute_5[112];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 112; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 112; ++i0_i1_fused_i2_fused_1) {
    compute_5[i0_i1_fused_i2_fused_1] = expf(ph_0[i0_i1_fused_i2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 112; ++i0_i1_fused_i2_fused_2) {
    compute_1[i0_i1_fused_i2_fused_2] = acoshf(compute_5[i0_i1_fused_i2_fused_2]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_3 = 0; i0_i1_fused_i2_fused_3 < 112; ++i0_i1_fused_i2_fused_3) {
    compute_2[i0_i1_fused_i2_fused_3] = asinf(compute_5[i0_i1_fused_i2_fused_3]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 8; ++i0) {
    for (int32_t i1 = 0; i1 < 2; ++i1) {
      for (int32_t i2 = 0; i2 < 7; ++i2) {
        compute_3[(((i0 * 14) + (i1 * 7)) + i2)] = fabsf((ph_0[(((i0 * 14) + (i1 * 7)) + i2)] - ph_3[(((i0 * 14) + (i1 * 7)) + i2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_4 = 0; i0_i1_fused_i2_fused_4 < 112; ++i0_i1_fused_i2_fused_4) {
    compute_4[i0_i1_fused_i2_fused_4] = acosf((ph_0[i0_i1_fused_i2_fused_4] - ph_3[i0_i1_fused_i2_fused_4]));
  }
}


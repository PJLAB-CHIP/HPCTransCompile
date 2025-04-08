void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* compute_3, float* compute_4, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 720; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = sinf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 18; ++i0) {
    for (int32_t i1 = 0; i1 < 10; ++i1) {
      for (int32_t i2 = 0; i2 < 4; ++i2) {
        compute_1[(((i0 * 40) + (i1 * 4)) + i2)] = fabsf(asinf(ph_0[(((i0 * 40) + (i1 * 4)) + i2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 720; ++i0_i1_fused_i2_fused_1) {
    compute_2[i0_i1_fused_i2_fused_1] = atanhf(asinf(ph_0[i0_i1_fused_i2_fused_1]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 720; ++i0_i1_fused_i2_fused_2) {
    compute_3[i0_i1_fused_i2_fused_2] = atanhf((ph_0[i0_i1_fused_i2_fused_2] * ph_3[i0_i1_fused_i2_fused_2]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 180; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 4; ++i2_1) {
      compute_4[((i0_i1_fused * 4) + i2_1)] = sinf((ph_0[((i0_i1_fused * 4) + i2_1)] * ph_3[((i0_i1_fused * 4) + i2_1)]));
    }
  }
}


void default_function_kernel(float* T_subtract, float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 20; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 17; ++i2) {
      compute[((i0_i1_fused * 17) + i2)] = atanf(ph_0[((i0_i1_fused * 17) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 10; ++ax0) {
    float compute_3[34];
    for (int32_t i1 = 0; i1 < 2; ++i1) {
      for (int32_t i2_1 = 0; i2_1 < 17; ++i2_1) {
        compute_3[((i1 * 17) + i2_1)] = expf(fabsf(ph_0[(((ax0 * 34) + (i1 * 17)) + i2_1)]));
      }
    }
    for (int32_t ax1 = 0; ax1 < 2; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 17; ++ax2) {
        T_subtract[(((ax0 * 34) + (ax1 * 17)) + ax2)] = (ph_0[(((ax0 * 34) + (ax1 * 17)) + ax2)] - compute_3[((ax1 * 17) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 340; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = atanhf(asinf(ph_0[i0_i1_fused_i2_fused]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 20; ++i0_i1_fused_1) {
    for (int32_t i2_2 = 0; i2_2 < 17; ++i2_2) {
      compute_2[((i0_i1_fused_1 * 17) + i2_2)] = fabsf(asinf(ph_0[((i0_i1_fused_1 * 17) + i2_2)]));
    }
  }
}


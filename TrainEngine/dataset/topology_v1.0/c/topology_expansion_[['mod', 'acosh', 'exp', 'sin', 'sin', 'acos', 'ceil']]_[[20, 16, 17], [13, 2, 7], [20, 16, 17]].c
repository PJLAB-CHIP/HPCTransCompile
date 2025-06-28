void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* compute_3, float* compute_4, float* ph_0, float* ph_3) {
  float compute_5[5440];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 5440; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = acoshf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 5440; ++i0_i1_fused_i2_fused_1) {
    compute_5[i0_i1_fused_i2_fused_1] = expf(ph_0[i0_i1_fused_i2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 5440; ++i0_i1_fused_i2_fused_2) {
    compute_1[i0_i1_fused_i2_fused_2] = sinf(compute_5[i0_i1_fused_i2_fused_2]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 320; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 17; ++i2) {
      compute_2[((i0_i1_fused * 17) + i2)] = sinf(compute_5[((i0_i1_fused * 17) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_3 = 0; i0_i1_fused_i2_fused_3 < 5440; ++i0_i1_fused_i2_fused_3) {
    compute_3[i0_i1_fused_i2_fused_3] = acosf(fmodf(ph_0[i0_i1_fused_i2_fused_3], ph_3[i0_i1_fused_i2_fused_3]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_4 = 0; i0_i1_fused_i2_fused_4 < 5440; ++i0_i1_fused_i2_fused_4) {
    compute_4[i0_i1_fused_i2_fused_4] = ceilf(fmodf(ph_0[i0_i1_fused_i2_fused_4], ph_3[i0_i1_fused_i2_fused_4]));
  }
}


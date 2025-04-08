void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 340; ++i0_i1_fused) {
    compute[i0_i1_fused] = atanhf(ph_0[i0_i1_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 340; ++i0_i1_fused_1) {
    compute_1[i0_i1_fused_1] = atanhf(sinf(ph_0[i0_i1_fused_1]));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 340; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = ceilf(ph_0[i0_i1_fused_i2_fused]);
  }
}


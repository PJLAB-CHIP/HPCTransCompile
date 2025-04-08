void default_function_kernel(float* T_mod, float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 3; ++i0) {
    for (int32_t i2 = 0; i2 < 18; ++i2) {
      compute[((i0 * 18) + i2)] = sinf((ph_0[((i0 * 18) + i2)] + sinf(ph_0[((i0 * 18) + i2)])));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 54; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = acosf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 3; ++ax0) {
    for (int32_t ax2 = 0; ax2 < 18; ++ax2) {
      T_mod[((ax0 * 18) + ax2)] = fmodf(atanhf(ph_0[((ax0 * 18) + ax2)]), ph_0[((ax0 * 18) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 54; ++i0_i1_fused_i2_fused_1) {
    compute_2[i0_i1_fused_i2_fused_1] = acosf(atanhf(ph_0[i0_i1_fused_i2_fused_1]));
  }
}


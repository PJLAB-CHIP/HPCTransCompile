void default_function_kernel(float* T_multiply, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 76; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = atanf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 76; ++i0_i1_fused) {
    compute_1[i0_i1_fused] = acoshf(asinhf(ph_0[i0_i1_fused]));
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 19; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 4; ++ax1) {
      T_multiply[((ax0 * 4) + ax1)] = (ph_0[((ax0 * 4) + ax1)] * acosf(ph_0[((ax0 * 4) + ax1)]));
    }
  }
}


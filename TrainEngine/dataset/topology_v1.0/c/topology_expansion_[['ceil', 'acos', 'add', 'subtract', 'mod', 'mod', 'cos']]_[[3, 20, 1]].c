void default_function_kernel(float* T_mod, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 60; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = ceilf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 3; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 20; ++ax1) {
      T_mod[((ax0 * 20) + ax1)] = fmodf(fmodf(ph_0[((ax0 * 20) + ax1)], (0.000000e+00f - acosf(ph_0[((ax0 * 20) + ax1)]))), ph_0[((ax0 * 20) + ax1)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 60; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = cosf(fmodf(ph_0[i0_i1_fused_i2_fused_1], (0.000000e+00f - acosf(ph_0[i0_i1_fused_i2_fused_1]))));
  }
}


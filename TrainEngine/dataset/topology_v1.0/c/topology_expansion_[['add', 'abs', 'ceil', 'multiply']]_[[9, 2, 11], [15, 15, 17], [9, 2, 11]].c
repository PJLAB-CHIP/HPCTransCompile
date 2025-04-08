void default_function_kernel(float* T_add, float* T_multiply, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 18; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 11; ++ax2) {
      T_add[((ax0_ax1_fused * 11) + ax2)] = (ph_0[((ax0_ax1_fused * 11) + ax2)] + ph_3[((ax0_ax1_fused * 11) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 198; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = fabsf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 9; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 2; ++ax1) {
      for (int32_t ax2_1 = 0; ax2_1 < 11; ++ax2_1) {
        T_multiply[(((ax0 * 22) + (ax1 * 11)) + ax2_1)] = (ceilf(ph_0[(((ax0 * 22) + (ax1 * 11)) + ax2_1)]) * ph_0[(((ax0 * 22) + (ax1 * 11)) + ax2_1)]);
      }
    }
  }
}


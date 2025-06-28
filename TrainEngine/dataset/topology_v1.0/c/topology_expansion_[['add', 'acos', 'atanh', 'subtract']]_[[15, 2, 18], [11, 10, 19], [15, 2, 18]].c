void default_function_kernel(float* T_add, float* T_subtract, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 540; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] + ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 15; ++i0) {
    for (int32_t i1 = 0; i1 < 2; ++i1) {
      for (int32_t i2 = 0; i2 < 18; ++i2) {
        compute[(((i0 * 36) + (i1 * 18)) + i2)] = acosf(ph_0[(((i0 * 36) + (i1 * 18)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 30; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 18; ++ax2) {
      T_subtract[((ax0_ax1_fused * 18) + ax2)] = (atanhf(ph_0[((ax0_ax1_fused * 18) + ax2)]) - ph_0[((ax0_ax1_fused * 18) + ax2)]);
    }
  }
}


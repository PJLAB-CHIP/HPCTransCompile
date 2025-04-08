void default_function_kernel(float* T_multiply, float* T_subtract, float* T_subtract_1, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 11; ++i0) {
    for (int32_t i1 = 0; i1 < 12; ++i1) {
      for (int32_t i2 = 0; i2 < 4; ++i2) {
        compute[(((i0 * 48) + (i1 * 4)) + i2)] = atanf(ph_0[(((i0 * 48) + (i1 * 4)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 132; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 4; ++ax2) {
      T_multiply[((ax0_ax1_fused * 4) + ax2)] = (ph_0[((ax0_ax1_fused * 4) + ax2)] * ceilf(atanhf(ph_0[((ax0_ax1_fused * 4) + ax2)])));
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 528; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (atanf(ph_0[ax0_ax1_fused_ax2_fused]) - ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 528; ++ax0_ax1_fused_ax2_fused_1) {
    T_subtract_1[ax0_ax1_fused_ax2_fused_1] = (atanf(ph_0[ax0_ax1_fused_ax2_fused_1]) - ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
}


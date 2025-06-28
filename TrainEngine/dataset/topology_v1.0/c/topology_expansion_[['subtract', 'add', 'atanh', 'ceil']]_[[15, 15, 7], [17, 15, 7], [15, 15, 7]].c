void default_function_kernel(float* T_add, float* T_subtract, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1575; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] - ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 15; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 15; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 7; ++ax2) {
        T_add[(((ax0 * 105) + (ax1 * 7)) + ax2)] = (ph_0[(((ax0 * 105) + (ax1 * 7)) + ax2)] + ph_3[(((ax0 * 105) + (ax1 * 7)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 15; ++i0) {
    for (int32_t i1 = 0; i1 < 15; ++i1) {
      for (int32_t i2 = 0; i2 < 7; ++i2) {
        compute[(((i0 * 105) + (i1 * 7)) + i2)] = ceilf(atanhf(ph_0[(((i0 * 105) + (i1 * 7)) + i2)]));
      }
    }
  }
}


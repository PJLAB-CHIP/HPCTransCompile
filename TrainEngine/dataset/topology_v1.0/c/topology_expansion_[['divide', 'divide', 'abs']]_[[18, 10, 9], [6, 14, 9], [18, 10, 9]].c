void default_function_kernel(float* T_divide, float* T_divide_1, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 18; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 9; ++ax2) {
        T_divide[(((ax0 * 90) + (ax1 * 9)) + ax2)] = (ph_0[(((ax0 * 90) + (ax1 * 9)) + ax2)] / ph_3[(((ax0 * 90) + (ax1 * 9)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1620; ++ax0_ax1_fused_ax2_fused) {
    T_divide_1[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] / ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 18; ++i0) {
    for (int32_t i1 = 0; i1 < 10; ++i1) {
      for (int32_t i2 = 0; i2 < 9; ++i2) {
        compute[(((i0 * 90) + (i1 * 9)) + i2)] = fabsf(ph_0[(((i0 * 90) + (i1 * 9)) + i2)]);
      }
    }
  }
}


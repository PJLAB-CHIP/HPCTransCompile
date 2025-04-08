void default_function_kernel(float* T_add, float* T_multiply, float* T_multiply_1, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 459; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = ((ph_0[ax0_ax1_fused_ax2_fused] * asinf(ph_0[ax0_ax1_fused_ax2_fused])) * ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 459; ++ax0_ax1_fused_ax2_fused_1) {
    T_add[ax0_ax1_fused_ax2_fused_1] = ((ph_0[ax0_ax1_fused_ax2_fused_1] * asinf(ph_0[ax0_ax1_fused_ax2_fused_1])) + ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 459; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf(fmodf(ph_0[i0_i1_fused_i2_fused], ph_3[i0_i1_fused_i2_fused]));
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 17; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 3; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 9; ++ax2) {
        T_multiply_1[(((ax0 * 27) + (ax1 * 9)) + ax2)] = (fmodf(ph_0[(((ax0 * 27) + (ax1 * 9)) + ax2)], ph_3[(((ax0 * 27) + (ax1 * 9)) + ax2)]) * ph_0[(((ax0 * 27) + (ax1 * 9)) + ax2)]);
      }
    }
  }
}


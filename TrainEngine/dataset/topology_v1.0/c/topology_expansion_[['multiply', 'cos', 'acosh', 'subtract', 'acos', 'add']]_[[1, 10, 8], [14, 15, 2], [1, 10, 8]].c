void default_function_kernel(float* T_add, float* T_subtract, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  for (int32_t i1 = 0; i1 < 10; ++i1) {
    for (int32_t i2 = 0; i2 < 8; ++i2) {
      compute[((i1 * 8) + i2)] = cosf(ph_0[((i1 * 8) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 80; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (acoshf(ph_0[ax0_ax1_fused_ax2_fused]) - ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 80; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = acosf(acoshf(ph_0[i0_i1_fused_i2_fused]));
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 10; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      T_add[((ax0_ax1_fused * 8) + ax2)] = ((ph_0[((ax0_ax1_fused * 8) + ax2)] * ph_3[((ax0_ax1_fused * 8) + ax2)]) + ph_0[((ax0_ax1_fused * 8) + ax2)]);
    }
  }
}


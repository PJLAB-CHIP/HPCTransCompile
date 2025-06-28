void default_function_kernel(float* T_add, float* T_divide, float* T_divide_1, float* compute, float* ph_0, float* ph_3) {
  for (int32_t ax1 = 0; ax1 < 12; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 13; ++ax2) {
      T_add[((ax1 * 13) + ax2)] = (ph_0[((ax1 * 13) + ax2)] + ph_3[((ax1 * 13) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 156; ++ax0_ax1_fused_ax2_fused) {
    T_divide[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] / ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 156; ++ax0_ax1_fused_ax2_fused_1) {
    T_divide_1[ax0_ax1_fused_ax2_fused_1] = (sinf(ph_0[ax0_ax1_fused_ax2_fused_1]) / ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
  for (int32_t i1 = 0; i1 < 12; ++i1) {
    for (int32_t i2 = 0; i2 < 13; ++i2) {
      compute[((i1 * 13) + i2)] = expf(sinf(ph_0[((i1 * 13) + i2)]));
    }
  }
}


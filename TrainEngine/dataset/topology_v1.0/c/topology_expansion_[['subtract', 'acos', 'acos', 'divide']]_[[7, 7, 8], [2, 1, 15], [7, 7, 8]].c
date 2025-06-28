void default_function_kernel(float* T_divide, float* T_subtract, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 49; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      T_subtract[((ax0_ax1_fused * 8) + ax2)] = (ph_0[((ax0_ax1_fused * 8) + ax2)] - ph_3[((ax0_ax1_fused * 8) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 392; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = acosf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < 49; ++ax0_ax1_fused_1) {
    for (int32_t ax2_1 = 0; ax2_1 < 8; ++ax2_1) {
      T_divide[((ax0_ax1_fused_1 * 8) + ax2_1)] = (acosf(ph_0[((ax0_ax1_fused_1 * 8) + ax2_1)]) / ph_0[((ax0_ax1_fused_1 * 8) + ax2_1)]);
    }
  }
}


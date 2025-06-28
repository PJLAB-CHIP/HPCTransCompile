void default_function_kernel(float* T_multiply, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 16; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = (fmodf(ph_0[ax0_ax1_fused_ax2_fused], acosf(ph_0[ax0_ax1_fused_ax2_fused])) * ph_0[ax0_ax1_fused_ax2_fused]);
  }
  for (int32_t i1 = 0; i1 < 2; ++i1) {
    for (int32_t i2 = 0; i2 < 8; ++i2) {
      compute[((i1 * 8) + i2)] = atanhf(ph_0[((i1 * 8) + i2)]);
    }
  }
  for (int32_t i1_1 = 0; i1_1 < 2; ++i1_1) {
    for (int32_t i2_1 = 0; i2_1 < 8; ++i2_1) {
      compute_1[((i1_1 * 8) + i2_1)] = atanf(ceilf(ph_0[((i1_1 * 8) + i2_1)]));
    }
  }
}


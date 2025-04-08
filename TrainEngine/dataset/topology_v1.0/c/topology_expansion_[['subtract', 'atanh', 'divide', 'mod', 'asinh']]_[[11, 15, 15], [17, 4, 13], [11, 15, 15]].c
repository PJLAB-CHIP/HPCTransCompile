void default_function_kernel(float* T_mod, float* T_subtract, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 2475; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] - ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 2475; ++ax0_ax1_fused_ax2_fused_1) {
    T_mod[ax0_ax1_fused_ax2_fused_1] = fmodf((ph_0[ax0_ax1_fused_ax2_fused_1] / atanhf(ph_0[ax0_ax1_fused_ax2_fused_1])), ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 11; ++i0) {
    for (int32_t i1 = 0; i1 < 15; ++i1) {
      for (int32_t i2 = 0; i2 < 15; ++i2) {
        compute[(((i0 * 225) + (i1 * 15)) + i2)] = asinhf((ph_0[(((i0 * 225) + (i1 * 15)) + i2)] / atanhf(ph_0[(((i0 * 225) + (i1 * 15)) + i2)])));
      }
    }
  }
}


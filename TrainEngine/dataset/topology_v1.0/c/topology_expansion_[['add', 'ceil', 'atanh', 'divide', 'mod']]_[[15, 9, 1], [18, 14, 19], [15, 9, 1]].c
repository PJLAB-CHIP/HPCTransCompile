void default_function_kernel(float* T_add, float* T_divide, float* T_mod, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 135; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] + ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 135; ++i0_i1_fused) {
    compute[i0_i1_fused] = ceilf(ph_0[i0_i1_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 135; ++ax0_ax1_fused_ax2_fused_1) {
    T_divide[ax0_ax1_fused_ax2_fused_1] = (atanhf(ph_0[ax0_ax1_fused_ax2_fused_1]) / ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 15; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 9; ++ax1) {
      T_mod[((ax0 * 9) + ax1)] = fmodf(atanhf(ph_0[((ax0 * 9) + ax1)]), ph_0[((ax0 * 9) + ax1)]);
    }
  }
}


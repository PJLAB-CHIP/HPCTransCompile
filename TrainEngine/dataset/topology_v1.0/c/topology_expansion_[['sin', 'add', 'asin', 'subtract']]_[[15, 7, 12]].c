void default_function_kernel(float* T_subtract, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1260; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] - asinf((ph_0[ax0_ax1_fused_ax2_fused] + sinf(ph_0[ax0_ax1_fused_ax2_fused]))));
  }
}


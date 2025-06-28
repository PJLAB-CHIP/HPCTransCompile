void default_function_kernel(float* T_reverse_sequence, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 672; ++ax0_ax1_fused_ax2_fused) {
    T_reverse_sequence[ax0_ax1_fused_ax2_fused] = data[(((ax0_ax1_fused_ax2_fused % 84) + 588) - ((ax0_ax1_fused_ax2_fused / 84) * 84))];
  }
}


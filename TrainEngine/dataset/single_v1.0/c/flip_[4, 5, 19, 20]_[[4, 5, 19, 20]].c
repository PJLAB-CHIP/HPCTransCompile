void default_function_kernel(float* T_reverse_sequence, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 380; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3 = 0; ax3 < 20; ++ax3) {
      T_reverse_sequence[((ax0_ax1_fused_ax2_fused * 20) + ax3)] = data[(((((ax0_ax1_fused_ax2_fused % 95) * 20) + ax3) + 5700) - ((ax0_ax1_fused_ax2_fused / 95) * 1900))];
    }
  }
}


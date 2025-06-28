void default_function_kernel(float* T_reverse_sequence, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 240; ++ax0_ax1_fused) {
    for (int32_t ax3 = 0; ax3 < 3; ++ax3) {
      T_reverse_sequence[((ax0_ax1_fused * 3) + ax3)] = data[(((((ax0_ax1_fused % 12) * 3) + ax3) + 684) - ((ax0_ax1_fused / 12) * 36))];
    }
  }
}


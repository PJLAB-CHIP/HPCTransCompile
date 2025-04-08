void default_function_kernel(float* adaptive_pool_max, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_ax3_fused = 0; ax0_ax1_fused_ax2_fused_ax3_fused < 21760; ++ax0_ax1_fused_ax2_fused_ax3_fused) {
    adaptive_pool_max[ax0_ax1_fused_ax2_fused_ax3_fused] = -3.402823e+38f;
    for (int32_t rv0 = 0; rv0 < ((((((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 4) + 4) % 8) == 0) ? (((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 3) + 3) >> 1) : ((((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 3) + 3) >> 1) + 1)) - ((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 12) >> 3)); ++rv0) {
      for (int32_t rv1 = 0; rv1 < (((((((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 7) + 7) % 8) == 0) ? ((((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 7) + 7) >> 3) : (((((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 7) + 7) >> 3) + 1)) - (((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 7) >> 3)); ++rv1) {
        adaptive_pool_max[ax0_ax1_fused_ax2_fused_ax3_fused] = max(adaptive_pool_max[ax0_ax1_fused_ax2_fused_ax3_fused], data[((((((ax0_ax1_fused_ax2_fused_ax3_fused >> 6) * 84) + (((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 12) >> 3) * 7)) + (rv0 * 7)) + (((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 7) >> 3)) + rv1)]);
      }
    }
  }
}


void default_function_kernel(float* adaptive_pool_max, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_ax3_fused = 0; ax0_ax1_fused_ax2_fused_ax3_fused < 1344; ++ax0_ax1_fused_ax2_fused_ax3_fused) {
    adaptive_pool_max[ax0_ax1_fused_ax2_fused_ax3_fused] = -3.402823e+38f;
    for (int32_t rv0 = 0; rv0 < ((((((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 4) + 4) % 8) == 0) ? (((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 5) + 5) >> 1) : ((((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 5) + 5) >> 1) + 1)) - ((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 20) >> 3)); ++rv0) {
      for (int32_t rv1 = 0; rv1 < (((((((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 2) + 2) % 8) == 0) ? ((((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 9) + 9) >> 2) : (((((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 9) + 9) >> 2) + 1)) - (((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 18) >> 3)); ++rv1) {
        adaptive_pool_max[ax0_ax1_fused_ax2_fused_ax3_fused] = max(adaptive_pool_max[ax0_ax1_fused_ax2_fused_ax3_fused], data[((((((ax0_ax1_fused_ax2_fused_ax3_fused >> 6) * 360) + (((((ax0_ax1_fused_ax2_fused_ax3_fused & 63) >> 3) * 20) >> 3) * 18)) + (rv0 * 18)) + (((ax0_ax1_fused_ax2_fused_ax3_fused & 7) * 18) >> 3)) + rv1)]);
      }
    }
  }
}


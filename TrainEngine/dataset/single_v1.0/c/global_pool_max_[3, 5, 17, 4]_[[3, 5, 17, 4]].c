void default_function_kernel(float* adaptive_pool_max, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 15; ++ax0_ax1_fused_ax2_fused) {
    adaptive_pool_max[ax0_ax1_fused_ax2_fused] = -3.402823e+38f;
    for (int32_t rv0 = 0; rv0 < 17; ++rv0) {
      for (int32_t rv1 = 0; rv1 < 4; ++rv1) {
        adaptive_pool_max[ax0_ax1_fused_ax2_fused] = max(adaptive_pool_max[ax0_ax1_fused_ax2_fused], data[(((ax0_ax1_fused_ax2_fused * 68) + (rv0 * 4)) + rv1)]);
      }
    }
  }
}


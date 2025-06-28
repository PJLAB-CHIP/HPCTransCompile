void default_function_kernel(float* adaptive_pool_avg, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 168; ++ax0_ax1_fused) {
    float adaptive_pool_sum[8];
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 8; ++ax3) {
        adaptive_pool_sum[ax3] = 0.000000e+00f;
        for (int32_t rv1 = 0; rv1 < (((((ax3 + 1) % 8) == 0) ? (((ax3 * 17) + 17) >> 3) : ((((ax3 * 17) + 17) >> 3) + 1)) - (ax3 * 2)); ++rv1) {
          adaptive_pool_sum[ax3] = (adaptive_pool_sum[ax3] + data[((((ax0_ax1_fused * 136) + (ax2 * 17)) + (ax3 * 2)) + rv1)]);
        }
      }
      for (int32_t ax3_1 = 0; ax3_1 < 8; ++ax3_1) {
        adaptive_pool_avg[(((ax0_ax1_fused * 64) + (ax2 * 8)) + ax3_1)] = (adaptive_pool_sum[ax3_1] / ((float)(((((ax3_1 + 1) % 8) == 0) ? (((ax3_1 * 17) + 17) >> 3) : ((((ax3_1 * 17) + 17) >> 3) + 1)) - (ax3_1 * 2))));
      }
    }
  }
}


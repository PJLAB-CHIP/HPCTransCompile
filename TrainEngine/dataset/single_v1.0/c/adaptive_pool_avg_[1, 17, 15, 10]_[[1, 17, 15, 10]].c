void default_function_kernel(float* adaptive_pool_avg, float* data) {
  float adaptive_pool_sum[8];
  for (int32_t ax1 = 0; ax1 < 17; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 8; ++ax3) {
        adaptive_pool_sum[ax3] = 0.000000e+00f;
        for (int32_t rv0 = 0; rv0 < ((((((ax2 * 7) + 7) % 8) == 0) ? (((ax2 * 15) + 15) >> 3) : ((((ax2 * 15) + 15) >> 3) + 1)) - ((ax2 * 15) >> 3)); ++rv0) {
          for (int32_t rv1 = 0; rv1 < ((((((ax3 * 2) + 2) % 8) == 0) ? (((ax3 * 5) + 5) >> 2) : ((((ax3 * 5) + 5) >> 2) + 1)) - ((ax3 * 10) >> 3)); ++rv1) {
            adaptive_pool_sum[ax3] = (adaptive_pool_sum[ax3] + data[(((((ax1 * 150) + (((ax2 * 15) >> 3) * 10)) + (rv0 * 10)) + ((ax3 * 10) >> 3)) + rv1)]);
          }
        }
      }
      for (int32_t ax3_1 = 0; ax3_1 < 8; ++ax3_1) {
        adaptive_pool_avg[(((ax1 * 64) + (ax2 * 8)) + ax3_1)] = (adaptive_pool_sum[ax3_1] / (((float)((((((ax2 * 7) + 7) % 8) == 0) ? (((ax2 * 15) + 15) >> 3) : ((((ax2 * 15) + 15) >> 3) + 1)) - ((ax2 * 15) >> 3))) * ((float)((((((ax3_1 * 2) + 2) % 8) == 0) ? (((ax3_1 * 5) + 5) >> 2) : ((((ax3_1 * 5) + 5) >> 2) + 1)) - ((ax3_1 * 10) >> 3)))));
      }
    }
  }
}


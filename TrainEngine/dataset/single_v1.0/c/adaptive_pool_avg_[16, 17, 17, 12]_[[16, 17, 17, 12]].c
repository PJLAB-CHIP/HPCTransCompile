void default_function_kernel(float* adaptive_pool_avg, float* data) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 16; ++ax0) {
    float adaptive_pool_sum[64];
    for (int32_t ax1 = 0; ax1 < 17; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
        for (int32_t ax3 = 0; ax3 < 8; ++ax3) {
          adaptive_pool_sum[((ax2 * 8) + ax3)] = 0.000000e+00f;
          for (int32_t rv0 = 0; rv0 < (((((ax2 + 1) % 8) == 0) ? (((ax2 * 17) + 17) >> 3) : ((((ax2 * 17) + 17) >> 3) + 1)) - (ax2 * 2)); ++rv0) {
            for (int32_t rv1 = 0; rv1 < ((((((ax3 * 4) + 4) % 8) == 0) ? (((ax3 * 3) + 3) >> 1) : ((((ax3 * 3) + 3) >> 1) + 1)) - ((ax3 * 12) >> 3)); ++rv1) {
              adaptive_pool_sum[((ax2 * 8) + ax3)] = (adaptive_pool_sum[((ax2 * 8) + ax3)] + data[((((((ax0 * 3468) + (ax1 * 204)) + (ax2 * 24)) + (rv0 * 12)) + ((ax3 * 12) >> 3)) + rv1)]);
            }
          }
        }
      }
      for (int32_t ax2_1 = 0; ax2_1 < 8; ++ax2_1) {
        for (int32_t ax3_1 = 0; ax3_1 < 8; ++ax3_1) {
          adaptive_pool_avg[((((ax0 * 1088) + (ax1 * 64)) + (ax2_1 * 8)) + ax3_1)] = (adaptive_pool_sum[((ax2_1 * 8) + ax3_1)] / (((float)(((((ax2_1 + 1) % 8) == 0) ? (((ax2_1 * 17) + 17) >> 3) : ((((ax2_1 * 17) + 17) >> 3) + 1)) - (ax2_1 * 2))) * ((float)((((((ax3_1 * 4) + 4) % 8) == 0) ? (((ax3_1 * 3) + 3) >> 1) : ((((ax3_1 * 3) + 3) >> 1) + 1)) - ((ax3_1 * 12) >> 3)))));
        }
      }
    }
  }
}


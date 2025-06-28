void default_function_kernel(float* adaptive_pool_avg, float* data) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 10; ++ax0) {
    float adaptive_pool_sum[1];
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 8; ++ax3) {
        adaptive_pool_sum[0] = 0.000000e+00f;
        for (int32_t rv0 = 0; rv0 < ((((((ax2 * 3) + 3) % 8) == 0) ? (((ax2 * 19) + 19) >> 3) : ((((ax2 * 19) + 19) >> 3) + 1)) - ((ax2 * 19) >> 3)); ++rv0) {
          for (int32_t rv1 = 0; rv1 < ((((((ax3 * 5) + 5) % 8) == 0) ? (((ax3 * 5) + 5) >> 3) : ((((ax3 * 5) + 5) >> 3) + 1)) - ((ax3 * 5) >> 3)); ++rv1) {
            adaptive_pool_sum[0] = (adaptive_pool_sum[0] + data[(((((ax0 * 95) + (((ax2 * 19) >> 3) * 5)) + (rv0 * 5)) + ((ax3 * 5) >> 3)) + rv1)]);
          }
        }
        adaptive_pool_avg[(((ax0 * 64) + (ax2 * 8)) + ax3)] = (adaptive_pool_sum[0] / (((float)((((((ax2 * 3) + 3) % 8) == 0) ? (((ax2 * 19) + 19) >> 3) : ((((ax2 * 19) + 19) >> 3) + 1)) - ((ax2 * 19) >> 3))) * ((float)((((((ax3 * 5) + 5) % 8) == 0) ? (((ax3 * 5) + 5) >> 3) : ((((ax3 * 5) + 5) >> 3) + 1)) - ((ax3 * 5) >> 3)))));
      }
    }
  }
}


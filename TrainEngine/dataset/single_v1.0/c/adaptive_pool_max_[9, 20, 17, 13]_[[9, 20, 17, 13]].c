void default_function_kernel(float* adaptive_pool_max, float* data) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 9; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 20; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
        for (int32_t ax3 = 0; ax3 < 8; ++ax3) {
          adaptive_pool_max[((((ax0 * 1280) + (ax1 * 64)) + (ax2 * 8)) + ax3)] = -3.402823e+38f;
          for (int32_t rv0 = 0; rv0 < (((((ax2 + 1) % 8) == 0) ? (((ax2 * 17) + 17) >> 3) : ((((ax2 * 17) + 17) >> 3) + 1)) - (ax2 * 2)); ++rv0) {
            for (int32_t rv1 = 0; rv1 < ((((((ax3 * 5) + 5) % 8) == 0) ? (((ax3 * 13) + 13) >> 3) : ((((ax3 * 13) + 13) >> 3) + 1)) - ((ax3 * 13) >> 3)); ++rv1) {
              adaptive_pool_max[((((ax0 * 1280) + (ax1 * 64)) + (ax2 * 8)) + ax3)] = max(adaptive_pool_max[((((ax0 * 1280) + (ax1 * 64)) + (ax2 * 8)) + ax3)], data[((((((ax0 * 4420) + (ax1 * 221)) + (ax2 * 26)) + (rv0 * 13)) + ((ax3 * 13) >> 3)) + rv1)]);
            }
          }
        }
      }
    }
  }
}


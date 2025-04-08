void default_function_kernel(float* adaptive_pool_max, float* data) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 13; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 19; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
        for (int32_t ax3 = 0; ax3 < 8; ++ax3) {
          adaptive_pool_max[((((ax0 * 1216) + (ax1 * 64)) + (ax2 * 8)) + ax3)] = -3.402823e+38f;
          for (int32_t rv0 = 0; rv0 < 2; ++rv0) {
            for (int32_t rv1 = 0; rv1 < ((((((ax3 * 3) + 3) % 8) == 0) ? (((ax3 * 3) + 3) >> 3) : ((((ax3 * 3) + 3) >> 3) + 1)) - ((ax3 * 3) >> 3)); ++rv1) {
              adaptive_pool_max[((((ax0 * 1216) + (ax1 * 64)) + (ax2 * 8)) + ax3)] = max(adaptive_pool_max[((((ax0 * 1216) + (ax1 * 64)) + (ax2 * 8)) + ax3)], data[((((((ax0 * 912) + (ax1 * 48)) + (ax2 * 6)) + (rv0 * 3)) + ((ax3 * 3) >> 3)) + rv1)]);
            }
          }
        }
      }
    }
  }
}


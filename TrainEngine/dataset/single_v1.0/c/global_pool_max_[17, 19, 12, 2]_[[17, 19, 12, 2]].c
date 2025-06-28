void default_function_kernel(float* adaptive_pool_max, float* data) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 17; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 19; ++ax1) {
      adaptive_pool_max[((ax0 * 19) + ax1)] = -3.402823e+38f;
      for (int32_t rv0 = 0; rv0 < 12; ++rv0) {
        for (int32_t rv1 = 0; rv1 < 2; ++rv1) {
          adaptive_pool_max[((ax0 * 19) + ax1)] = max(adaptive_pool_max[((ax0 * 19) + ax1)], data[((((ax0 * 456) + (ax1 * 24)) + (rv0 * 2)) + rv1)]);
        }
      }
    }
  }
}


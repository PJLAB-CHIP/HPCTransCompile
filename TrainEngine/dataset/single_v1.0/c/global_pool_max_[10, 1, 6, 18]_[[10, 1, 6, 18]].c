void default_function_kernel(float* adaptive_pool_max, float* data) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 10; ++ax0) {
    adaptive_pool_max[ax0] = -3.402823e+38f;
    for (int32_t rv0 = 0; rv0 < 6; ++rv0) {
      for (int32_t rv1 = 0; rv1 < 18; ++rv1) {
        adaptive_pool_max[ax0] = max(adaptive_pool_max[ax0], data[(((ax0 * 108) + (rv0 * 18)) + rv1)]);
      }
    }
  }
}


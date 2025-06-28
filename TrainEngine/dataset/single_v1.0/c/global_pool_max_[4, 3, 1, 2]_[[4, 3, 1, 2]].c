void default_function_kernel(float* adaptive_pool_max, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 12; ++ax0_ax1_fused) {
    adaptive_pool_max[ax0_ax1_fused] = -3.402823e+38f;
    for (int32_t rv1 = 0; rv1 < 2; ++rv1) {
      adaptive_pool_max[ax0_ax1_fused] = max(adaptive_pool_max[ax0_ax1_fused], data[((ax0_ax1_fused * 2) + rv1)]);
    }
  }
}


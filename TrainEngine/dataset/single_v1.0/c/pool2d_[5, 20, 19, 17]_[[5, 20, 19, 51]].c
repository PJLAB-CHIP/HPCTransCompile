void default_function_kernel(float* data, float* pool_max) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 100; ++ax0_ax1_fused) {
    float pad_temp[9];
    for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 26; ++ax3) {
        for (int32_t ax2_ax3_fused_s = 0; ax2_ax3_fused_s < 9; ++ax2_ax3_fused_s) {
          pad_temp[ax2_ax3_fused_s] = (((((1 <= ((ax2 * 2) + (ax2_ax3_fused_s / 3))) && (((ax2_ax3_fused_s / 6) + ax2) < 10)) && (1 <= ((ax3 * 2) + (ax2_ax3_fused_s % 3)))) && ((((ax2_ax3_fused_s % 3) >> 1) + ax3) < 26)) ? data[((((((ax0_ax1_fused * 969) + (ax2 * 102)) + ((ax2_ax3_fused_s / 3) * 51)) + (ax3 * 2)) + (ax2_ax3_fused_s % 3)) - 52)] : -3.402823e+38f);
        }
        pool_max[(((ax0_ax1_fused * 260) + (ax2 * 26)) + ax3)] = -3.402823e+38f;
        for (int32_t rv0 = 0; rv0 < 3; ++rv0) {
          for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
            pool_max[(((ax0_ax1_fused * 260) + (ax2 * 26)) + ax3)] = max(pool_max[(((ax0_ax1_fused * 260) + (ax2 * 26)) + ax3)], pad_temp[((rv0 * 3) + rv1)]);
          }
        }
      }
    }
  }
}


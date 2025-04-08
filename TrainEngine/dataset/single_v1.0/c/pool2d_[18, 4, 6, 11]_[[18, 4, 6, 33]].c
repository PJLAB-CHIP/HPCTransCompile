void default_function_kernel(float* data, float* pool_max) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 72; ++ax0_ax1_fused) {
    float pad_temp[9];
    for (int32_t ax2 = 0; ax2 < 3; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 17; ++ax3) {
        for (int32_t ax2_1 = 0; ax2_1 < 3; ++ax2_1) {
          for (int32_t ax3_s = 0; ax3_s < 3; ++ax3_s) {
            pad_temp[((ax2_1 * 3) + ax3_s)] = ((((1 <= ((ax2 * 2) + ax2_1)) && (1 <= ((ax3 * 2) + ax3_s))) && (((ax3_s >> 1) + ax3) < 17)) ? data[((((((ax0_ax1_fused * 198) + (ax2 * 66)) + (ax2_1 * 33)) + (ax3 * 2)) + ax3_s) - 34)] : -3.402823e+38f);
          }
        }
        pool_max[(((ax0_ax1_fused * 51) + (ax2 * 17)) + ax3)] = -3.402823e+38f;
        for (int32_t rv0 = 0; rv0 < 3; ++rv0) {
          for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
            pool_max[(((ax0_ax1_fused * 51) + (ax2 * 17)) + ax3)] = max(pool_max[(((ax0_ax1_fused * 51) + (ax2 * 17)) + ax3)], pad_temp[((rv0 * 3) + rv1)]);
          }
        }
      }
    }
  }
}


void default_function_kernel(float* data, float* pool_max) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 5; ++ax0) {
    float pad_temp[663];
    for (int32_t ax1 = 0; ax1 < 6; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 4; ++ax2) {
        for (int32_t ax2_1 = 0; ax2_1 < 3; ++ax2_1) {
          for (int32_t ax3 = 0; ax3 < 13; ++ax3) {
            for (int32_t ax4_s = 0; ax4_s < 17; ++ax4_s) {
              pad_temp[(((ax2_1 * 221) + (ax3 * 17)) + ax4_s)] = ((((((1 <= ((ax2 * 2) + ax2_1)) && (((ax2_1 >> 1) + ax2) < 4)) && (1 <= ax3)) && (ax3 < 12)) && (1 <= ax4_s)) ? data[(((((((ax0 * 7392) + (ax1 * 1232)) + (ax2 * 352)) + (ax2_1 * 176)) + (ax3 * 16)) + ax4_s) - 193)] : -3.402823e+38f);
            }
          }
        }
        for (int32_t ax3_1 = 0; ax3_1 < 6; ++ax3_1) {
          for (int32_t ax4 = 0; ax4 < 8; ++ax4) {
            pool_max[(((((ax0 * 1152) + (ax1 * 192)) + (ax2 * 48)) + (ax3_1 * 8)) + ax4)] = -3.402823e+38f;
            for (int32_t rv0 = 0; rv0 < 3; ++rv0) {
              for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
                for (int32_t rv2 = 0; rv2 < 3; ++rv2) {
                  pool_max[(((((ax0 * 1152) + (ax1 * 192)) + (ax2 * 48)) + (ax3_1 * 8)) + ax4)] = max(pool_max[(((((ax0 * 1152) + (ax1 * 192)) + (ax2 * 48)) + (ax3_1 * 8)) + ax4)], pad_temp[(((((rv0 * 221) + (ax3_1 * 34)) + (rv1 * 17)) + (ax4 * 2)) + rv2)]);
                }
              }
            }
          }
        }
      }
    }
  }
}


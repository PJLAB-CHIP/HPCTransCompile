void default_function_kernel(float* data, float* pool_max) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 10; ++ax0) {
    float pad_temp[153];
    for (int32_t ax1 = 0; ax1 < 8; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
        for (int32_t ax2_1 = 0; ax2_1 < 3; ++ax2_1) {
          for (int32_t ax3 = 0; ax3 < 3; ++ax3) {
            for (int32_t ax4 = 0; ax4 < 17; ++ax4) {
              pad_temp[(((ax2_1 * 51) + (ax3 * 17)) + ax4)] = (((((1 <= ((ax2 * 2) + ax2_1)) && (1 <= ax3)) && (ax3 < 2)) && (1 <= ax4)) ? data[(((((((ax0 * 2048) + (ax1 * 256)) + (ax2 * 32)) + (ax2_1 * 16)) + (ax3 * 16)) + ax4) - 33)] : -3.402823e+38f);
            }
          }
        }
        for (int32_t ax4_1 = 0; ax4_1 < 8; ++ax4_1) {
          pool_max[((((ax0 * 512) + (ax1 * 64)) + (ax2 * 8)) + ax4_1)] = -3.402823e+38f;
          for (int32_t rv0 = 0; rv0 < 3; ++rv0) {
            for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
              for (int32_t rv2 = 0; rv2 < 3; ++rv2) {
                pool_max[((((ax0 * 512) + (ax1 * 64)) + (ax2 * 8)) + ax4_1)] = max(pool_max[((((ax0 * 512) + (ax1 * 64)) + (ax2 * 8)) + ax4_1)], pad_temp[((((rv0 * 51) + (rv1 * 17)) + (ax4_1 * 2)) + rv2)]);
              }
            }
          }
        }
      }
    }
  }
}


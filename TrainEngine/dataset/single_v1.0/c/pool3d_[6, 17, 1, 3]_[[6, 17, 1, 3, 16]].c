void default_function_kernel(float* data, float* pool_max) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_ax3_fused = 0; ax0_ax1_fused_ax2_fused_ax3_fused < 204; ++ax0_ax1_fused_ax2_fused_ax3_fused) {
    float pad_temp[153];
    for (int32_t ax2 = 0; ax2 < 3; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 3; ++ax3) {
        for (int32_t ax4 = 0; ax4 < 17; ++ax4) {
          pad_temp[(((ax2 * 51) + (ax3 * 17)) + ax4)] = ((((((1 <= ax2) && (ax2 < 2)) && (1 <= (((ax0_ax1_fused_ax2_fused_ax3_fused & 1) * 2) + ax3))) && (((ax3 >> 1) + (ax0_ax1_fused_ax2_fused_ax3_fused & 1)) < 2)) && (1 <= ax4)) ? data[(((((((ax0_ax1_fused_ax2_fused_ax3_fused >> 1) * 48) + (ax2 * 48)) + ((ax0_ax1_fused_ax2_fused_ax3_fused & 1) * 32)) + (ax3 * 16)) + ax4) - 65)] : -3.402823e+38f);
        }
      }
    }
    for (int32_t ax4_1 = 0; ax4_1 < 8; ++ax4_1) {
      pool_max[((ax0_ax1_fused_ax2_fused_ax3_fused * 8) + ax4_1)] = -3.402823e+38f;
      for (int32_t rv0 = 0; rv0 < 3; ++rv0) {
        for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
          for (int32_t rv2 = 0; rv2 < 3; ++rv2) {
            pool_max[((ax0_ax1_fused_ax2_fused_ax3_fused * 8) + ax4_1)] = max(pool_max[((ax0_ax1_fused_ax2_fused_ax3_fused * 8) + ax4_1)], pad_temp[((((rv0 * 51) + (rv1 * 17)) + (ax4_1 * 2)) + rv2)]);
          }
        }
      }
    }
  }
}


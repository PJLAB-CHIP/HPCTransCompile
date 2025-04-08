void default_function_kernel(float* data, float* pool_max) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_ax3_fused = 0; ax0_ax1_fused_ax2_fused_ax3_fused < 104; ++ax0_ax1_fused_ax2_fused_ax3_fused) {
    float pad_temp[27];
    for (int32_t ax4 = 0; ax4 < 8; ++ax4) {
      for (int32_t ax2 = 0; ax2 < 3; ++ax2) {
        for (int32_t ax3 = 0; ax3 < 3; ++ax3) {
          for (int32_t ax4_s = 0; ax4_s < 3; ++ax4_s) {
            pad_temp[(((ax2 * 9) + (ax3 * 3)) + ax4_s)] = (((((1 <= (((ax0_ax1_fused_ax2_fused_ax3_fused & 1) * 2) + ax2)) && (((ax2 >> 1) + (ax0_ax1_fused_ax2_fused_ax3_fused & 1)) < 2)) && (1 <= ax3)) && (1 <= ((ax4 * 2) + ax4_s))) ? data[((((((((ax0_ax1_fused_ax2_fused_ax3_fused >> 1) * 96) + ((ax0_ax1_fused_ax2_fused_ax3_fused & 1) * 64)) + (ax2 * 32)) + (ax3 * 16)) + (ax4 * 2)) + ax4_s) - 49)] : -3.402823e+38f);
          }
        }
      }
      pool_max[((ax0_ax1_fused_ax2_fused_ax3_fused * 8) + ax4)] = -3.402823e+38f;
      for (int32_t rv0 = 0; rv0 < 3; ++rv0) {
        for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
          for (int32_t rv2 = 0; rv2 < 3; ++rv2) {
            pool_max[((ax0_ax1_fused_ax2_fused_ax3_fused * 8) + ax4)] = max(pool_max[((ax0_ax1_fused_ax2_fused_ax3_fused * 8) + ax4)], pad_temp[(((rv0 * 9) + (rv1 * 3)) + rv2)]);
          }
        }
      }
    }
  }
}


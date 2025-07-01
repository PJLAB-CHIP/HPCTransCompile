void default_function_kernel(float* A, float* T_gather_nd, float* indices) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 8; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3 = 0; ax3 < 64; ++ax3) {
      for (int32_t ax4_outer = 0; ax4_outer < 4; ++ax4_outer) {
        for (int32_t ax4_inner = 0; ax4_inner < 16; ++ax4_inner) {
          if (((ax4_outer * 8) + (ax4_inner >> 1)) < 25) {
            T_gather_nd[((((ax0_ax1_fused_ax2_fused * 3200) + (ax3 * 50)) + (ax4_outer * 16)) + ax4_inner)] = A[(((((((int32_t)indices[ax0_ax1_fused_ax2_fused]) * 204800) + (((int32_t)indices[(ax0_ax1_fused_ax2_fused + 8)]) * 3200)) + (ax3 * 50)) + (ax4_outer * 16)) + ax4_inner)];
          }
        }
      }
    }
  }
}


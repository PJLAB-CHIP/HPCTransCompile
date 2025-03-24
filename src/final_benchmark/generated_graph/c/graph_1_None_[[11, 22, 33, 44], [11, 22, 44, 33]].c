void default_function_kernel(float* T_transpose, float* ph, float* ph_1) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 363; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 22; ++ax2) {
      for (int32_t ax3_outer = 0; ax3_outer < 3; ++ax3_outer) {
        for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
          if (((ax3_outer * 4) + (ax3_inner >> 2)) < 11) {
            T_transpose[((((ax0_ax1_fused * 968) + (ax2 * 44)) + (ax3_outer * 16)) + ax3_inner)] = (ph[((((((ax0_ax1_fused % 11) * 31944) + ((((ax3_outer * 8) + (ax3_inner >> 1)) % 22) * 1452)) + ((ax0_ax1_fused / 11) * 44)) + ((ax3_inner & 1) * 22)) + ax2)] + ph_1[(((((((ax0_ax1_fused % 121) / 11) * 31944) + (ax2 * 1452)) + ((((ax3_outer * 16) + ax3_inner) % 44) * 33)) + ((ax0_ax1_fused % 11) * 3)) + (ax0_ax1_fused / 121))]);
          }
        }
      }
    }
  }
}


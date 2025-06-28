void default_function_kernel(float* T_strided_slice, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 216; ++ax0_ax1_fused_ax2_fused) {
    float T_reshape[20];
    if (0 <= (((ax0_ax1_fused_ax2_fused % 72) >> 1) - (((ax0_ax1_fused_ax2_fused % 72) >> 2) * 2))) {
      for (int32_t ax5 = 0; ax5 < 20; ++ax5) {
        T_reshape[ax5] = data[((((ax0_ax1_fused_ax2_fused & 3) * 1080) + ((ax0_ax1_fused_ax2_fused >> 2) * 20)) + ax5)];
      }
    }
    for (int32_t ax3 = 0; ax3 < 20; ++ax3) {
      T_strided_slice[((ax0_ax1_fused_ax2_fused * 20) + ax3)] = T_reshape[ax3];
    }
  }
}


void default_function_kernel(float* T_cast, float* data, float* weight) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 624; ++ax0_ax1_fused_ax2_fused) {
    float T_multiply_red[2];
    for (int32_t ax2 = 0; ax2 < 2; ++ax2) {
      T_multiply_red[ax2] = 0.000000e+00f;
      for (int32_t k1 = 0; k1 < 16; ++k1) {
        T_multiply_red[ax2] = (T_multiply_red[ax2] + (data[(((((ax0_ax1_fused_ax2_fused / 208) * 416) + (k1 * 26)) + ((ax0_ax1_fused_ax2_fused % 13) * 2)) + ax2)] * data[(((((ax0_ax1_fused_ax2_fused / 208) * 416) + (k1 * 26)) + ((ax0_ax1_fused_ax2_fused % 13) * 2)) + ax2)]));
      }
    }
    for (int32_t ax3_s = 0; ax3_s < 2; ++ax3_s) {
      T_cast[((ax0_ax1_fused_ax2_fused * 2) + ax3_s)] = ((data[((ax0_ax1_fused_ax2_fused * 2) + ax3_s)] * weight[((ax0_ax1_fused_ax2_fused % 208) / 13)]) * (1.000000e+00f / sqrtf(((T_multiply_red[ax3_s] * 6.250000e-02f) + 1.000000e-05f))));
    }
  }
}


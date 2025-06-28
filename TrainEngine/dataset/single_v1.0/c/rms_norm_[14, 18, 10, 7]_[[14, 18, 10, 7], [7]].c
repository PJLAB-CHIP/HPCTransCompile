void default_function_kernel(float* T_cast, float* data, float* weight) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 252; ++ax0_ax1_fused) {
    float T_multiply_red[70];
    for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 7; ++ax2) {
        T_multiply_red[((ax1 * 7) + ax2)] = 0.000000e+00f;
        for (int32_t k1 = 0; k1 < 18; ++k1) {
          T_multiply_red[((ax1 * 7) + ax2)] = (T_multiply_red[((ax1 * 7) + ax2)] + (data[(((((ax0_ax1_fused / 18) * 1260) + (k1 * 70)) + (ax1 * 7)) + ax2)] * data[(((((ax0_ax1_fused / 18) * 1260) + (k1 * 70)) + (ax1 * 7)) + ax2)]));
        }
      }
    }
    for (int32_t ax2_1 = 0; ax2_1 < 10; ++ax2_1) {
      for (int32_t ax3_s = 0; ax3_s < 7; ++ax3_s) {
        T_cast[(((ax0_ax1_fused * 70) + (ax2_1 * 7)) + ax3_s)] = ((data[(((ax0_ax1_fused * 70) + (ax2_1 * 7)) + ax3_s)] * weight[(ax0_ax1_fused % 18)]) * (1.000000e+00f / sqrtf(((T_multiply_red[((ax2_1 * 7) + ax3_s)] * 5.555556e-02f) + 1.000000e-05f))));
      }
    }
  }
}


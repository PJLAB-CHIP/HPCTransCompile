void default_function_kernel(float* T_cast, float* data, float* weight) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 4; ++ax0_ax1_fused) {
    float T_multiply_red[36];
    for (int32_t ax1 = 0; ax1 < 4; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 9; ++ax2) {
        T_multiply_red[((ax1 * 9) + ax2)] = 0.000000e+00f;
        for (int32_t k1 = 0; k1 < 2; ++k1) {
          T_multiply_red[((ax1 * 9) + ax2)] = (T_multiply_red[((ax1 * 9) + ax2)] + (data[(((((ax0_ax1_fused >> 1) * 72) + (k1 * 36)) + (ax1 * 9)) + ax2)] * data[(((((ax0_ax1_fused >> 1) * 72) + (k1 * 36)) + (ax1 * 9)) + ax2)]));
        }
      }
    }
    for (int32_t ax2_1 = 0; ax2_1 < 4; ++ax2_1) {
      for (int32_t ax3_s = 0; ax3_s < 9; ++ax3_s) {
        T_cast[(((ax0_ax1_fused * 36) + (ax2_1 * 9)) + ax3_s)] = ((data[(((ax0_ax1_fused * 36) + (ax2_1 * 9)) + ax3_s)] * weight[(ax0_ax1_fused & 1)]) * (1.000000e+00f / sqrtf(((T_multiply_red[((ax2_1 * 9) + ax3_s)] * 5.000000e-01f) + 1.000000e-05f))));
      }
    }
  }
}


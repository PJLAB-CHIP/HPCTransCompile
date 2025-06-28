void default_function_kernel(float* T_cast, float* data, float* weight) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 8; ++ax0) {
    float T_multiply_red[168];
    for (int32_t ax1 = 0; ax1 < 12; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 14; ++ax2) {
        T_multiply_red[((ax1 * 14) + ax2)] = 0.000000e+00f;
        for (int32_t k1 = 0; k1 < 10; ++k1) {
          T_multiply_red[((ax1 * 14) + ax2)] = (T_multiply_red[((ax1 * 14) + ax2)] + (data[((((ax0 * 1680) + (k1 * 168)) + (ax1 * 14)) + ax2)] * data[((((ax0 * 1680) + (k1 * 168)) + (ax1 * 14)) + ax2)]));
        }
      }
    }
    for (int32_t ax1_1 = 0; ax1_1 < 10; ++ax1_1) {
      for (int32_t ax2_1 = 0; ax2_1 < 12; ++ax2_1) {
        for (int32_t ax3_s = 0; ax3_s < 14; ++ax3_s) {
          T_cast[((((ax0 * 1680) + (ax1_1 * 168)) + (ax2_1 * 14)) + ax3_s)] = ((data[((((ax0 * 1680) + (ax1_1 * 168)) + (ax2_1 * 14)) + ax3_s)] * weight[ax1_1]) * (1.000000e+00f / sqrtf(((T_multiply_red[((ax2_1 * 14) + ax3_s)] * 1.000000e-01f) + 1.000000e-05f))));
        }
      }
    }
  }
}


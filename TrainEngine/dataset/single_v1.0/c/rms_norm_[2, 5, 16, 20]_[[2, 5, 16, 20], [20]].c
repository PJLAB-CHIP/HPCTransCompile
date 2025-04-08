void default_function_kernel(float* T_cast, float* data, float* weight) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 2; ++ax0) {
    float T_multiply_red[320];
    for (int32_t ax1 = 0; ax1 < 16; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 20; ++ax2) {
        T_multiply_red[((ax1 * 20) + ax2)] = 0.000000e+00f;
        for (int32_t k1 = 0; k1 < 5; ++k1) {
          T_multiply_red[((ax1 * 20) + ax2)] = (T_multiply_red[((ax1 * 20) + ax2)] + (data[((((ax0 * 1600) + (k1 * 320)) + (ax1 * 20)) + ax2)] * data[((((ax0 * 1600) + (k1 * 320)) + (ax1 * 20)) + ax2)]));
        }
      }
    }
    for (int32_t ax1_1 = 0; ax1_1 < 5; ++ax1_1) {
      for (int32_t ax2_1 = 0; ax2_1 < 16; ++ax2_1) {
        for (int32_t ax3 = 0; ax3 < 20; ++ax3) {
          T_cast[((((ax0 * 1600) + (ax1_1 * 320)) + (ax2_1 * 20)) + ax3)] = ((data[((((ax0 * 1600) + (ax1_1 * 320)) + (ax2_1 * 20)) + ax3)] * weight[ax1_1]) * (1.000000e+00f / sqrtf(((T_multiply_red[((ax2_1 * 20) + ax3)] * 2.000000e-01f) + 1.000000e-05f))));
        }
      }
    }
  }
}


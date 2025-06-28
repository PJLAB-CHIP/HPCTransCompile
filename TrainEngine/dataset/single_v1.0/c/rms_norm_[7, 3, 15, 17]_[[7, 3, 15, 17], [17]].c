void default_function_kernel(float* T_cast, float* data, float* weight) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 7; ++ax0) {
    float T_multiply_red[1];
    for (int32_t ax1 = 0; ax1 < 3; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 15; ++ax2) {
        for (int32_t ax3 = 0; ax3 < 17; ++ax3) {
          T_multiply_red[0] = 0.000000e+00f;
          for (int32_t k1 = 0; k1 < 3; ++k1) {
            T_multiply_red[0] = (T_multiply_red[0] + (data[((((ax0 * 765) + (k1 * 255)) + (ax2 * 17)) + ax3)] * data[((((ax0 * 765) + (k1 * 255)) + (ax2 * 17)) + ax3)]));
          }
          T_cast[((((ax0 * 765) + (ax1 * 255)) + (ax2 * 17)) + ax3)] = ((data[((((ax0 * 765) + (ax1 * 255)) + (ax2 * 17)) + ax3)] * weight[ax1]) * (1.000000e+00f / sqrtf(((T_multiply_red[0] * 3.333333e-01f) + 1.000000e-05f))));
        }
      }
    }
  }
}


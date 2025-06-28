void default_function_kernel(float* T_divide, float* data, float* moving_mean, float* moving_var) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 3; ++ax0) {
    float T_reshape[11];
    float T_reshape_1[1];
    for (int32_t ax1 = 0; ax1 < 11; ++ax1) {
      T_reshape[ax1] = moving_mean[ax1];
    }
    for (int32_t ax1_1 = 0; ax1_1 < 11; ++ax1_1) {
      for (int32_t ax2 = 0; ax2 < 2; ++ax2) {
        for (int32_t ax3 = 0; ax3 < 16; ++ax3) {
          T_reshape_1[0] = moving_var[ax1_1];
          T_divide[((((ax0 * 352) + (ax1_1 * 32)) + (ax2 * 16)) + ax3)] = ((data[((((ax0 * 352) + (ax1_1 * 32)) + (ax2 * 16)) + ax3)] - T_reshape[ax1_1]) / sqrtf((T_reshape_1[0] + 1.000000e-05f)));
        }
      }
    }
  }
}


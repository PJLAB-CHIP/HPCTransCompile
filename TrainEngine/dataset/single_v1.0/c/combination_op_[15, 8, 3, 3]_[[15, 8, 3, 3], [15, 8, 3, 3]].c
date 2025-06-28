void default_function_kernel(float* T_add, float* data, float* data_1) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 15; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 8; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 3; ++ax2) {
        for (int32_t ax3 = 0; ax3 < 3; ++ax3) {
          T_add[((((ax0 * 72) + (ax1 * 9)) + (ax2 * 3)) + ax3)] = (sqrtf(data[((((ax0 * 72) + (ax1 * 9)) + (ax2 * 3)) + ax3)]) + cosf(data_1[((((ax0 * 72) + (ax1 * 9)) + (ax2 * 3)) + ax3)]));
        }
      }
    }
  }
}


void default_function_kernel(float* T_add, float* data, float* data_1) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 12; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
      for (int32_t ax3 = 0; ax3 < 5; ++ax3) {
        T_add[(((ax0 * 50) + (ax1 * 5)) + ax3)] = (data[(((ax0 * 50) + (ax1 * 5)) + ax3)] + data_1[(((ax0 * 50) + (ax1 * 5)) + ax3)]);
      }
    }
  }
}


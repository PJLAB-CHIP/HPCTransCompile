void default_function_kernel(float* T_strided_slice, float* a) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 3; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 5; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 7; ++ax2) {
        T_strided_slice[(((ax0 * 35) + (ax1 * 7)) + ax2)] = a[((((ax0 * 168) + (ax1 * 12)) + ax2) + 195)];
      }
    }
  }
}


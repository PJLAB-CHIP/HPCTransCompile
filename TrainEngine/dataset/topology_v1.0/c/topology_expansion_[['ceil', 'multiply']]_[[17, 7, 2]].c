void default_function_kernel(float* T_multiply, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 17; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 7; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 2; ++ax2) {
        T_multiply[(((ax0 * 14) + (ax1 * 2)) + ax2)] = (ph_0[(((ax0 * 14) + (ax1 * 2)) + ax2)] * ceilf(ph_0[(((ax0 * 14) + (ax1 * 2)) + ax2)]));
      }
    }
  }
}


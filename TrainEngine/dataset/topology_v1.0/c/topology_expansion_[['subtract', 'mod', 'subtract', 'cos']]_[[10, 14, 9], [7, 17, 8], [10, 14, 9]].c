void default_function_kernel(float* T_subtract, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 10; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 14; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 9; ++ax2) {
        T_subtract[(((ax0 * 126) + (ax1 * 9)) + ax2)] = (ph_0[(((ax0 * 126) + (ax1 * 9)) + ax2)] - ph_3[(((ax0 * 126) + (ax1 * 9)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 10; ++i0) {
    for (int32_t i1 = 0; i1 < 14; ++i1) {
      for (int32_t i2 = 0; i2 < 9; ++i2) {
        compute[(((i0 * 126) + (i1 * 9)) + i2)] = cosf((ph_0[(((i0 * 126) + (i1 * 9)) + i2)] - fmodf(ph_0[(((i0 * 126) + (i1 * 9)) + i2)], ph_3[(((i0 * 126) + (i1 * 9)) + i2)])));
      }
    }
  }
}


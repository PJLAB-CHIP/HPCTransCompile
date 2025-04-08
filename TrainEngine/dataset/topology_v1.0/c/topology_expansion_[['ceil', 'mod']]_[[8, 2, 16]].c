void default_function_kernel(float* T_mod, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 8; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 2; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 16; ++ax2) {
        T_mod[(((ax0 * 32) + (ax1 * 16)) + ax2)] = fmodf(ph_0[(((ax0 * 32) + (ax1 * 16)) + ax2)], ceilf(ph_0[(((ax0 * 32) + (ax1 * 16)) + ax2)]));
      }
    }
  }
}


void default_function_kernel(float* T_divide, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 15; ++i0) {
    for (int32_t i1 = 0; i1 < 20; ++i1) {
      for (int32_t i2 = 0; i2 < 17; ++i2) {
        compute[(((i0 * 340) + (i1 * 17)) + i2)] = cosf(ph_0[(((i0 * 340) + (i1 * 17)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 15; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 20; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 17; ++ax2) {
        T_divide[(((ax0 * 340) + (ax1 * 17)) + ax2)] = ((ph_0[(((ax0 * 340) + (ax1 * 17)) + ax2)] * (ph_0[(((ax0 * 340) + (ax1 * 17)) + ax2)] / asinf(atanf(ph_0[(((ax0 * 340) + (ax1 * 17)) + ax2)])))) / ph_0[(((ax0 * 340) + (ax1 * 17)) + ax2)]);
      }
    }
  }
}


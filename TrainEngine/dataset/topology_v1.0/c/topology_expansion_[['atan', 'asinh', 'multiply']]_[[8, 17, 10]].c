void default_function_kernel(float* T_multiply, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 136; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 10; ++i2) {
      compute[((i0_i1_fused * 10) + i2)] = atanf(ph_0[((i0_i1_fused * 10) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 8; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 17; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
        T_multiply[(((ax0 * 170) + (ax1 * 10)) + ax2)] = (asinhf(ph_0[(((ax0 * 170) + (ax1 * 10)) + ax2)]) * ph_0[(((ax0 * 170) + (ax1 * 10)) + ax2)]);
      }
    }
  }
}


void default_function_kernel(float* T_subtract, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 7; ++ax0) {
    float compute_1[1];
    for (int32_t ax2 = 0; ax2 < 18; ++ax2) {
      compute_1[0] = expf(ph_0[((ax0 * 18) + ax2)]);
      T_subtract[((ax0 * 18) + ax2)] = ((ph_0[((ax0 * 18) + ax2)] - compute_1[0]) - ph_0[((ax0 * 18) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 7; ++i0) {
    for (int32_t i2 = 0; i2 < 18; ++i2) {
      compute[((i0 * 18) + i2)] = asinf(ph_0[((i0 * 18) + i2)]);
    }
  }
}


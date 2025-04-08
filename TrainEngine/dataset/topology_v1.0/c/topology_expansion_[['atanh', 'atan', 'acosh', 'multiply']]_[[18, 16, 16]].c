void default_function_kernel(float* T_multiply, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 288; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 16; ++i2) {
      compute[((i0_i1_fused * 16) + i2)] = atanhf(ph_0[((i0_i1_fused * 16) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 18; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 16; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 16; ++ax2) {
        T_multiply[(((ax0 * 256) + (ax1 * 16)) + ax2)] = (ph_0[(((ax0 * 256) + (ax1 * 16)) + ax2)] * acoshf(atanf(ph_0[(((ax0 * 256) + (ax1 * 16)) + ax2)])));
      }
    }
  }
}


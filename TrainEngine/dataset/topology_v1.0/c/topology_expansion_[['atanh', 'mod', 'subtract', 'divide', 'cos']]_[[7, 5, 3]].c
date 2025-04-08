void default_function_kernel(float* T_divide, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 7; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 5; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 3; ++ax2) {
        T_divide[(((ax0 * 15) + (ax1 * 3)) + ax2)] = (ph_0[(((ax0 * 15) + (ax1 * 3)) + ax2)] / (fmodf(ph_0[(((ax0 * 15) + (ax1 * 3)) + ax2)], atanhf(ph_0[(((ax0 * 15) + (ax1 * 3)) + ax2)])) - ph_0[(((ax0 * 15) + (ax1 * 3)) + ax2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 35; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 3; ++i2) {
      compute[((i0_i1_fused * 3) + i2)] = cosf(ph_0[((i0_i1_fused * 3) + i2)]);
    }
  }
}


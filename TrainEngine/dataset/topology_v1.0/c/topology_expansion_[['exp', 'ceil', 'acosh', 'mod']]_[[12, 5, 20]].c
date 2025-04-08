void default_function_kernel(float* T_mod, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 12; ++i0) {
    for (int32_t i1 = 0; i1 < 5; ++i1) {
      for (int32_t i2 = 0; i2 < 20; ++i2) {
        compute[(((i0 * 100) + (i1 * 20)) + i2)] = expf(ph_0[(((i0 * 100) + (i1 * 20)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 60; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 20; ++ax2) {
      T_mod[((ax0_ax1_fused * 20) + ax2)] = fmodf(ph_0[((ax0_ax1_fused * 20) + ax2)], acoshf(ceilf(ph_0[((ax0_ax1_fused * 20) + ax2)])));
    }
  }
}


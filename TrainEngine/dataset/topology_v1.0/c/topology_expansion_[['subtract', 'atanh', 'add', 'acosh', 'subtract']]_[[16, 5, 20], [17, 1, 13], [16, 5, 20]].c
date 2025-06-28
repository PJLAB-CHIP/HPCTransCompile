void default_function_kernel(float* T_subtract, float* T_subtract_1, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 16; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 5; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 20; ++ax2) {
        T_subtract[(((ax0 * 100) + (ax1 * 20)) + ax2)] = (ph_0[(((ax0 * 100) + (ax1 * 20)) + ax2)] - ph_3[(((ax0 * 100) + (ax1 * 20)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 16; ++i0) {
    for (int32_t i1 = 0; i1 < 5; ++i1) {
      for (int32_t i2 = 0; i2 < 20; ++i2) {
        compute[(((i0 * 100) + (i1 * 20)) + i2)] = acoshf((ph_0[(((i0 * 100) + (i1 * 20)) + i2)] + atanhf(ph_0[(((i0 * 100) + (i1 * 20)) + i2)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1600; ++ax0_ax1_fused_ax2_fused) {
    T_subtract_1[ax0_ax1_fused_ax2_fused] = atanhf(ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


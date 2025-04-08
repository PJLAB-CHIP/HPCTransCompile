void default_function_kernel(float* T_divide, float* T_mod, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1056; ++ax0_ax1_fused_ax2_fused) {
    T_divide[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] / ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 6; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 16; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 11; ++ax2) {
        T_mod[(((ax0 * 176) + (ax1 * 11)) + ax2)] = fmodf(ph_0[(((ax0 * 176) + (ax1 * 11)) + ax2)], ph_3[(((ax0 * 176) + (ax1 * 11)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 96; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 11; ++i2) {
      compute[((i0_i1_fused * 11) + i2)] = acoshf(ph_0[((i0_i1_fused * 11) + i2)]);
    }
  }
}


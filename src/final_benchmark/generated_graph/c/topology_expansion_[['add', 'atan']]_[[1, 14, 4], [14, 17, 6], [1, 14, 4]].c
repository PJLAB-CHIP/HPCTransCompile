void default_function_kernel(float* T_add, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 14; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 4; ++ax2) {
      T_add[((ax0_ax1_fused * 4) + ax2)] = (ph_0[((ax0_ax1_fused * 4) + ax2)] + ph_3[((ax0_ax1_fused * 4) + ax2)]);
    }
  }
  for (int32_t i1 = 0; i1 < 14; ++i1) {
    for (int32_t i2 = 0; i2 < 4; ++i2) {
      compute[((i1 * 4) + i2)] = atanf(ph_0[((i1 * 4) + i2)]);
    }
  }
}


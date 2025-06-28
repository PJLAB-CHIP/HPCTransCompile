void default_function_kernel(float* T_subtract, float* ph_0) {
  float compute[4788];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 4788; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 14; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 19; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 18; ++ax2) {
        T_subtract[(((ax0 * 342) + (ax1 * 18)) + ax2)] = (ph_0[(((ax0 * 342) + (ax1 * 18)) + ax2)] - compute[(((ax0 * 342) + (ax1 * 18)) + ax2)]);
      }
    }
  }
}


void default_function_kernel(float* T_multiply, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 576; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = fabsf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 72; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      T_multiply[((ax0_ax1_fused * 8) + ax2)] = (atanhf(ph_0[((ax0_ax1_fused * 8) + ax2)]) * ph_0[((ax0_ax1_fused * 8) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 8; ++i0) {
    for (int32_t i1 = 0; i1 < 9; ++i1) {
      for (int32_t i2 = 0; i2 < 8; ++i2) {
        compute_1[(((i0 * 72) + (i1 * 8)) + i2)] = acoshf(ph_0[(((i0 * 72) + (i1 * 8)) + i2)]);
      }
    }
  }
}


void default_function_kernel(float* T_subtract, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 3458; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] - ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 14; ++i0) {
    for (int32_t i1 = 0; i1 < 19; ++i1) {
      for (int32_t i2 = 0; i2 < 13; ++i2) {
        compute[(((i0 * 247) + (i1 * 13)) + i2)] = sinf(ph_0[(((i0 * 247) + (i1 * 13)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 266; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 13; ++i2_1) {
      compute_1[((i0_i1_fused * 13) + i2_1)] = cosf(ceilf(ph_0[((i0_i1_fused * 13) + i2_1)]));
    }
  }
}


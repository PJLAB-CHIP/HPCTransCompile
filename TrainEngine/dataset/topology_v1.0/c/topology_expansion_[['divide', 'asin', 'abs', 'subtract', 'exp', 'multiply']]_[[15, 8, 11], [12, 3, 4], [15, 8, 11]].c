void default_function_kernel(float* T_multiply, float* T_subtract, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1320; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 15; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 8; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 11; ++ax2) {
        T_subtract[(((ax0 * 88) + (ax1 * 11)) + ax2)] = (fabsf(ph_0[(((ax0 * 88) + (ax1 * 11)) + ax2)]) - ph_0[(((ax0 * 88) + (ax1 * 11)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 120; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 11; ++i2) {
      compute_1[((i0_i1_fused * 11) + i2)] = expf(fabsf(ph_0[((i0_i1_fused * 11) + i2)]));
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1320; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = ((ph_0[ax0_ax1_fused_ax2_fused] / ph_3[ax0_ax1_fused_ax2_fused]) * ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


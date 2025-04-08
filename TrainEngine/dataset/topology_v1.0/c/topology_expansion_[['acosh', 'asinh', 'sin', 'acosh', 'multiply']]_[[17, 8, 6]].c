void default_function_kernel(float* T_multiply, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 17; ++i0) {
    for (int32_t i1 = 0; i1 < 8; ++i1) {
      for (int32_t i2 = 0; i2 < 6; ++i2) {
        compute[(((i0 * 48) + (i1 * 6)) + i2)] = acoshf(ph_0[(((i0 * 48) + (i1 * 6)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 136; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 6; ++i2_1) {
      compute_1[((i0_i1_fused * 6) + i2_1)] = sinf(asinhf(ph_0[((i0_i1_fused * 6) + i2_1)]));
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 816; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] * acoshf(ph_0[ax0_ax1_fused_ax2_fused]));
  }
}


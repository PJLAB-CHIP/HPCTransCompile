void default_function_kernel(float* T_multiply, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 20; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] * ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 10; ++i0) {
    for (int32_t i1 = 0; i1 < 2; ++i1) {
      compute[((i0 * 2) + i1)] = atanhf(ph_0[((i0 * 2) + i1)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 10; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 2; ++i1_1) {
      compute_1[((i0_1 * 2) + i1_1)] = ceilf(ph_0[((i0_1 * 2) + i1_1)]);
    }
  }
}


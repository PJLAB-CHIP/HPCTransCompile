void default_function_kernel(float* T_add, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 19; ++i0) {
    for (int32_t i1 = 0; i1 < 7; ++i1) {
      for (int32_t i2 = 0; i2 < 20; ++i2) {
        compute[(((i0 * 140) + (i1 * 20)) + i2)] = fabsf(ph_0[(((i0 * 140) + (i1 * 20)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 133; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 20; ++i2_1) {
      compute_1[((i0_i1_fused * 20) + i2_1)] = fabsf(atanhf(ph_0[((i0_i1_fused * 20) + i2_1)]));
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 2660; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] + acosf(ph_0[ax0_ax1_fused_ax2_fused]));
  }
}


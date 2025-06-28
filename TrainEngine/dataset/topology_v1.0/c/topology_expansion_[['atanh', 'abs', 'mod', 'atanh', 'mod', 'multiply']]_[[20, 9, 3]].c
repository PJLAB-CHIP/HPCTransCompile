void default_function_kernel(float* T_mod, float* T_multiply, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 180; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 3; ++i2) {
      compute[((i0_i1_fused * 3) + i2)] = atanhf(ph_0[((i0_i1_fused * 3) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 180; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 3; ++ax2) {
      T_mod[((ax0_ax1_fused * 3) + ax2)] = fmodf(fabsf(ph_0[((ax0_ax1_fused * 3) + ax2)]), ph_0[((ax0_ax1_fused * 3) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 540; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = (fmodf(ph_0[ax0_ax1_fused_ax2_fused], atanhf(ph_0[ax0_ax1_fused_ax2_fused])) * ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


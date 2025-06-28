void default_function_kernel(float* T_subtract, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 15; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 2; ++i2) {
      compute[((i0_i1_fused * 2) + i2)] = fabsf(ph_0[((i0_i1_fused * 2) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 30; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] - asinhf(cosf(ph_0[ax0_ax1_fused_ax2_fused])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 15; ++i0_i1_fused_1) {
    for (int32_t i2_1 = 0; i2_1 < 2; ++i2_1) {
      compute_1[((i0_i1_fused_1 * 2) + i2_1)] = asinf(atanhf(ph_0[((i0_i1_fused_1 * 2) + i2_1)]));
    }
  }
}


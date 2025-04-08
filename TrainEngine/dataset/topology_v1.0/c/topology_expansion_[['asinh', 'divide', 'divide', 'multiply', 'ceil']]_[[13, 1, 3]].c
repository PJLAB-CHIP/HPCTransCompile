void default_function_kernel(float* T_multiply, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 39; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] * ((ph_0[ax0_ax1_fused_ax2_fused] / asinhf(ph_0[ax0_ax1_fused_ax2_fused])) / ph_0[ax0_ax1_fused_ax2_fused]));
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 13; ++i0) {
    for (int32_t i2 = 0; i2 < 3; ++i2) {
      compute[((i0 * 3) + i2)] = ceilf(ph_0[((i0 * 3) + i2)]);
    }
  }
}


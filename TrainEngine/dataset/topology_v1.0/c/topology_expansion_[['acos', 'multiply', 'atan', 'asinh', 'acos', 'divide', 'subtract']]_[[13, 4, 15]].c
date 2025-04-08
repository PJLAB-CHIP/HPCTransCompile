void default_function_kernel(float* T_divide, float* T_subtract, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 780; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = atanf((ph_0[i0_i1_fused_i2_fused] * acosf(ph_0[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 780; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = asinhf(ph_0[i0_i1_fused_i2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 52; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 15; ++ax2) {
      T_divide[((ax0_ax1_fused * 15) + ax2)] = (acosf(ph_0[((ax0_ax1_fused * 15) + ax2)]) / ph_0[((ax0_ax1_fused * 15) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 780; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (acosf(ph_0[ax0_ax1_fused_ax2_fused]) - ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


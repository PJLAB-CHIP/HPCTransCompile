void default_function_kernel(float* T_add, float* T_divide, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 4; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 6; ++ax2) {
        T_divide[(((ax0 * 60) + (ax1 * 6)) + ax2)] = ((ph_0[(((ax0 * 60) + (ax1 * 6)) + ax2)] - fabsf(ph_0[(((ax0 * 60) + (ax1 * 6)) + ax2)])) / ph_0[(((ax0 * 60) + (ax1 * 6)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 240; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = ((ph_0[ax0_ax1_fused_ax2_fused] - fabsf(ph_0[ax0_ax1_fused_ax2_fused])) + ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 240; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = fabsf((ph_0[i0_i1_fused_i2_fused] - ph_3[i0_i1_fused_i2_fused]));
  }
}


void default_function_kernel(float* T_mod, float* T_mod_1, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 612; ++ax0_ax1_fused_ax2_fused) {
    T_mod[ax0_ax1_fused_ax2_fused] = fmodf((ph_0[ax0_ax1_fused_ax2_fused] / asinhf(ph_0[ax0_ax1_fused_ax2_fused])), ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 612; ++ax0_ax1_fused_ax2_fused_1) {
    T_mod_1[ax0_ax1_fused_ax2_fused_1] = fmodf((ph_0[ax0_ax1_fused_ax2_fused_1] / asinhf(ph_0[ax0_ax1_fused_ax2_fused_1])), ph_0[ax0_ax1_fused_ax2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 36; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 17; ++i2) {
      compute[((i0_i1_fused * 17) + i2)] = cosf((ph_0[((i0_i1_fused * 17) + i2)] * ph_3[((i0_i1_fused * 17) + i2)]));
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 9; ++i0) {
    for (int32_t i1 = 0; i1 < 4; ++i1) {
      for (int32_t i2_1 = 0; i2_1 < 17; ++i2_1) {
        compute_1[(((i0 * 68) + (i1 * 17)) + i2_1)] = fabsf((ph_0[(((i0 * 68) + (i1 * 17)) + i2_1)] * ph_3[(((i0 * 68) + (i1 * 17)) + i2_1)]));
      }
    }
  }
}


void default_function_kernel(float* T_mod, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 4160; ++ax0_ax1_fused_ax2_fused) {
    T_mod[ax0_ax1_fused_ax2_fused] = fmodf((ph_0[ax0_ax1_fused_ax2_fused] * atanhf(ph_0[ax0_ax1_fused_ax2_fused])), ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 13; ++i0) {
    for (int32_t i1 = 0; i1 < 20; ++i1) {
      for (int32_t i2 = 0; i2 < 16; ++i2) {
        compute[(((i0 * 320) + (i1 * 16)) + i2)] = fabsf((ph_0[(((i0 * 320) + (i1 * 16)) + i2)] * atanhf(ph_0[(((i0 * 320) + (i1 * 16)) + i2)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 260; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 16; ++i2_1) {
      compute_1[((i0_i1_fused * 16) + i2_1)] = sinf((ph_0[((i0_i1_fused * 16) + i2_1)] * ph_3[((i0_i1_fused * 16) + i2_1)]));
    }
  }
}


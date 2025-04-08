void default_function_kernel(float* T_mod, float* compute, float* compute_1, float* compute_2, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 96; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 5; ++i2) {
      compute[((i0_i1_fused * 5) + i2)] = atanf(sinf(ph_0[((i0_i1_fused * 5) + i2)]));
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 12; ++i0) {
    for (int32_t i1 = 0; i1 < 8; ++i1) {
      for (int32_t i2_1 = 0; i2_1 < 5; ++i2_1) {
        compute_1[(((i0 * 40) + (i1 * 5)) + i2_1)] = ceilf(sinf(ph_0[(((i0 * 40) + (i1 * 5)) + i2_1)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 480; ++ax0_ax1_fused_ax2_fused) {
    T_mod[ax0_ax1_fused_ax2_fused] = fmodf(ph_0[ax0_ax1_fused_ax2_fused], ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 480; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = sinf((ph_0[i0_i1_fused_i2_fused] / ph_3[i0_i1_fused_i2_fused]));
  }
}


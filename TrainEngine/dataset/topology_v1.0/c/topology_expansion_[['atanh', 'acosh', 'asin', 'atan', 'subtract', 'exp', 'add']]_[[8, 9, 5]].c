void default_function_kernel(float* T_add, float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 8; ++i0) {
    for (int32_t i1 = 0; i1 < 9; ++i1) {
      for (int32_t i2 = 0; i2 < 5; ++i2) {
        compute[(((i0 * 45) + (i1 * 5)) + i2)] = atanhf(ph_0[(((i0 * 45) + (i1 * 5)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 72; ++i0_i1_fused) {
    for (int32_t i2_1 = 0; i2_1 < 5; ++i2_1) {
      compute_1[((i0_i1_fused * 5) + i2_1)] = asinf(acoshf(ph_0[((i0_i1_fused * 5) + i2_1)]));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 360; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = expf((ph_0[i0_i1_fused_i2_fused] - atanf(ph_0[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 360; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = ((ph_0[ax0_ax1_fused_ax2_fused] - atanf(ph_0[ax0_ax1_fused_ax2_fused])) + ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


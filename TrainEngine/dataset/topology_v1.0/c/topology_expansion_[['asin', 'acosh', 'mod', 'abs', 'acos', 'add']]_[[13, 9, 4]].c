void default_function_kernel(float* T_add, float* T_mod, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 468; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 468; ++ax0_ax1_fused_ax2_fused) {
    T_mod[ax0_ax1_fused_ax2_fused] = fmodf(acoshf(ph_0[ax0_ax1_fused_ax2_fused]), ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 13; ++i0) {
    for (int32_t i1 = 0; i1 < 9; ++i1) {
      for (int32_t i2 = 0; i2 < 4; ++i2) {
        compute_1[(((i0 * 36) + (i1 * 4)) + i2)] = fabsf(ph_0[(((i0 * 36) + (i1 * 4)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 13; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 9; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 4; ++ax2) {
        T_add[(((ax0 * 36) + (ax1 * 4)) + ax2)] = (acosf(ph_0[(((ax0 * 36) + (ax1 * 4)) + ax2)]) + ph_0[(((ax0 * 36) + (ax1 * 4)) + ax2)]);
      }
    }
  }
}


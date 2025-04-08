void default_function_kernel(float* T_subtract, float* compute, float* compute_1, float* compute_2, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 18; ++i0) {
    for (int32_t i1 = 0; i1 < 14; ++i1) {
      for (int32_t i2 = 0; i2 < 18; ++i2) {
        compute[(((i0 * 252) + (i1 * 18)) + i2)] = acoshf(acoshf(ph_0[(((i0 * 252) + (i1 * 18)) + i2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 18; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 14; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 18; ++i2_1) {
        compute_1[(((i0_1 * 252) + (i1_1 * 18)) + i2_1)] = expf(acoshf(ph_0[(((i0_1 * 252) + (i1_1 * 18)) + i2_1)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 4536; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] - ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 4536; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = acosf((ph_0[i0_i1_fused_i2_fused] * ph_3[i0_i1_fused_i2_fused]));
  }
}


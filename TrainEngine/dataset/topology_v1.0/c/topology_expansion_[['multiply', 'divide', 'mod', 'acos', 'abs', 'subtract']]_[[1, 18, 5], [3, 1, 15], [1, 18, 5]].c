void default_function_kernel(float* T_subtract, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  for (int32_t i1 = 0; i1 < 18; ++i1) {
    for (int32_t i2 = 0; i2 < 5; ++i2) {
      compute[((i1 * 5) + i2)] = acosf(fmodf(ph_0[((i1 * 5) + i2)], (ph_0[((i1 * 5) + i2)] / ph_3[((i1 * 5) + i2)])));
    }
  }
  for (int32_t i1_1 = 0; i1_1 < 18; ++i1_1) {
    for (int32_t i2_1 = 0; i2_1 < 5; ++i2_1) {
      compute_1[((i1_1 * 5) + i2_1)] = fabsf(fmodf(ph_0[((i1_1 * 5) + i2_1)], (ph_0[((i1_1 * 5) + i2_1)] / ph_3[((i1_1 * 5) + i2_1)])));
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 90; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = ((ph_0[ax0_ax1_fused_ax2_fused] * ph_3[ax0_ax1_fused_ax2_fused]) - ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


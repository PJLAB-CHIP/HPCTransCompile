void default_function_kernel(float* T_add, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 72; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 14; ++i2) {
      compute[((i0_i1_fused * 14) + i2)] = atanf(fmodf(ph_0[((i0_i1_fused * 14) + i2)], (ph_0[((i0_i1_fused * 14) + i2)] / ph_3[((i0_i1_fused * 14) + i2)])));
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 6; ++i0) {
    for (int32_t i1 = 0; i1 < 12; ++i1) {
      for (int32_t i2_1 = 0; i2_1 < 14; ++i2_1) {
        compute_1[(((i0 * 168) + (i1 * 14)) + i2_1)] = ceilf(fmodf(ph_0[(((i0 * 168) + (i1 * 14)) + i2_1)], (ph_0[(((i0 * 168) + (i1 * 14)) + i2_1)] / ph_3[(((i0 * 168) + (i1 * 14)) + i2_1)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1008; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = ((ph_0[ax0_ax1_fused_ax2_fused] - ph_3[ax0_ax1_fused_ax2_fused]) + ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


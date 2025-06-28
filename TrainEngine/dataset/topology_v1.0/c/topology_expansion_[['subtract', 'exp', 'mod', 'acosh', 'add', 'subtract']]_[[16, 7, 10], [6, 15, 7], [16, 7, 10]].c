void default_function_kernel(float* T_add, float* T_subtract, float* compute, float* ph_0, float* ph_3) {
  float compute_1[1120];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1120; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 16; ++i0) {
    for (int32_t i1 = 0; i1 < 7; ++i1) {
      for (int32_t i2 = 0; i2 < 10; ++i2) {
        compute[(((i0 * 70) + (i1 * 10)) + i2)] = acoshf(fmodf(ph_0[(((i0 * 70) + (i1 * 10)) + i2)], compute_1[(((i0 * 70) + (i1 * 10)) + i2)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 16; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 7; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
        T_add[(((ax0 * 70) + (ax1 * 10)) + ax2)] = (fmodf(ph_0[(((ax0 * 70) + (ax1 * 10)) + ax2)], compute_1[(((ax0 * 70) + (ax1 * 10)) + ax2)]) + ph_0[(((ax0 * 70) + (ax1 * 10)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1120; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = ((ph_0[ax0_ax1_fused_ax2_fused] - ph_3[ax0_ax1_fused_ax2_fused]) - ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


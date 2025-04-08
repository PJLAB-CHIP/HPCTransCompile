void default_function_kernel(float* T_divide, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 119; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 3; ++ax2) {
      T_divide[((ax0_ax1_fused * 3) + ax2)] = (ph_0[((ax0_ax1_fused * 3) + ax2)] / ph_3[((ax0_ax1_fused * 3) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 7; ++i0) {
    for (int32_t i1 = 0; i1 < 17; ++i1) {
      for (int32_t i2 = 0; i2 < 3; ++i2) {
        compute[(((i0 * 51) + (i1 * 3)) + i2)] = cosf(ph_0[(((i0 * 51) + (i1 * 3)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 357; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = fabsf(ph_0[i0_i1_fused_i2_fused]);
  }
}


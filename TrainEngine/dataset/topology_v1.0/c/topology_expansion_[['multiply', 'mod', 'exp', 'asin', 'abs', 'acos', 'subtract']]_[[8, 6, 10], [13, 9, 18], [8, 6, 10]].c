void default_function_kernel(float* T_mod, float* T_subtract, float* compute, float* compute_1, float* compute_2, float* ph_0, float* ph_3) {
  float compute_3[480];
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 480; ++ax0_ax1_fused_ax2_fused) {
    T_mod[ax0_ax1_fused_ax2_fused] = fmodf(ph_0[ax0_ax1_fused_ax2_fused], ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 48; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 10; ++i2) {
      compute_3[((i0_i1_fused * 10) + i2)] = expf(ph_0[((i0_i1_fused * 10) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 480; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf(compute_3[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 480; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = fabsf(compute_3[i0_i1_fused_i2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 480; ++i0_i1_fused_i2_fused_2) {
    compute_2[i0_i1_fused_i2_fused_2] = acosf((ph_0[i0_i1_fused_i2_fused_2] * ph_3[i0_i1_fused_i2_fused_2]));
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 8; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 6; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
        T_subtract[(((ax0 * 60) + (ax1 * 10)) + ax2)] = ((ph_0[(((ax0 * 60) + (ax1 * 10)) + ax2)] * ph_3[(((ax0 * 60) + (ax1 * 10)) + ax2)]) - ph_0[(((ax0 * 60) + (ax1 * 10)) + ax2)]);
      }
    }
  }
}


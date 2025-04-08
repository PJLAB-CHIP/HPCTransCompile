void default_function_kernel(float* T_multiply, float* T_multiply_1, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 143; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 2; ++ax2) {
      T_multiply[((ax0_ax1_fused * 2) + ax2)] = (ph_0[((ax0_ax1_fused * 2) + ax2)] * ph_3[((ax0_ax1_fused * 2) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < 143; ++ax0_ax1_fused_1) {
    for (int32_t ax2_1 = 0; ax2_1 < 2; ++ax2_1) {
      T_multiply_1[((ax0_ax1_fused_1 * 2) + ax2_1)] = (ph_0[((ax0_ax1_fused_1 * 2) + ax2_1)] * ph_3[((ax0_ax1_fused_1 * 2) + ax2_1)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 286; ++i0_i1_fused_i2_fused) {
    float compute_1[1];
    compute_1[0] = expf(ph_0[i0_i1_fused_i2_fused]);
    compute[i0_i1_fused_i2_fused] = sinf(compute_1[0]);
  }
}


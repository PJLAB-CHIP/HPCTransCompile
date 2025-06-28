void default_function_kernel(float* T_add, float* T_multiply, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 55; ++ax0_ax1_fused) {
    T_add[ax0_ax1_fused] = (ph_0[ax0_ax1_fused] + ph_3[ax0_ax1_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 55; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < 55; ++ax0_ax1_fused_1) {
    T_multiply[ax0_ax1_fused_1] = (atanhf(ph_0[ax0_ax1_fused_1]) * ph_0[ax0_ax1_fused_1]);
  }
}


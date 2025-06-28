void default_function_kernel(float* T_multiply, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 612; ++ax0_ax1_fused_ax2_fused) {
    T_multiply[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] * ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 612; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = fabsf((ph_0[i0_i1_fused_i2_fused] / (ph_0[i0_i1_fused_i2_fused] / ph_3[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 68; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 9; ++i2) {
      compute_1[((i0_i1_fused * 9) + i2)] = fabsf((ph_0[((i0_i1_fused * 9) + i2)] / (ph_0[((i0_i1_fused * 9) + i2)] / ph_3[((i0_i1_fused * 9) + i2)])));
    }
  }
}


void default_function_kernel(float* T_subtract, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 22; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] - ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 22; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinhf(fmodf(ph_0[i0_i1_fused_i2_fused], atanhf(ph_0[i0_i1_fused_i2_fused])));
  }
  for (int32_t i1 = 0; i1 < 11; ++i1) {
    for (int32_t i2 = 0; i2 < 2; ++i2) {
      compute_1[((i1 * 2) + i2)] = atanhf(fmodf(ph_0[((i1 * 2) + i2)], atanhf(ph_0[((i1 * 2) + i2)])));
    }
  }
}


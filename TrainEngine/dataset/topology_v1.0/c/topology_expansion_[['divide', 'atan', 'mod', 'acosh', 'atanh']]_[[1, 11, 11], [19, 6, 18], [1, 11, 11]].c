void default_function_kernel(float* T_divide, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 121; ++ax0_ax1_fused_ax2_fused) {
    T_divide[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] / ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 11; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 11; ++i2) {
      compute[((i0_i1_fused * 11) + i2)] = acoshf(fmodf(ph_0[((i0_i1_fused * 11) + i2)], atanf(ph_0[((i0_i1_fused * 11) + i2)])));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 121; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = atanhf(fmodf(ph_0[i0_i1_fused_i2_fused], atanf(ph_0[i0_i1_fused_i2_fused])));
  }
}


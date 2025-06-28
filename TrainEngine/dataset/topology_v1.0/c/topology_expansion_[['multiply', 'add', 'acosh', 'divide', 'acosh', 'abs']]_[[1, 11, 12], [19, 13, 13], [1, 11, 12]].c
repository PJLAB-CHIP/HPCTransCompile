void default_function_kernel(float* T_add, float* T_divide, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  for (int32_t ax1 = 0; ax1 < 11; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 12; ++ax2) {
      T_divide[((ax1 * 12) + ax2)] = (acoshf(ph_0[((ax1 * 12) + ax2)]) / ph_0[((ax1 * 12) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 132; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = acoshf(acoshf(ph_0[i0_i1_fused_i2_fused]));
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 132; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] + ph_3[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 132; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = fabsf((ph_0[i0_i1_fused_i2_fused_1] * ph_3[i0_i1_fused_i2_fused_1]));
  }
}


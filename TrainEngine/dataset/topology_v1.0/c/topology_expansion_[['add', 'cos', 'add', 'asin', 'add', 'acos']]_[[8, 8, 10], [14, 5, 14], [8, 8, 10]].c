void default_function_kernel(float* T_add, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 640; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf((ph_0[i0_i1_fused_i2_fused] + cosf(ph_0[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 64; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
      T_add[((ax0_ax1_fused * 10) + ax2)] = ((ph_0[((ax0_ax1_fused * 10) + ax2)] + cosf(ph_0[((ax0_ax1_fused * 10) + ax2)])) + ph_0[((ax0_ax1_fused * 10) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 640; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = acosf((ph_0[i0_i1_fused_i2_fused_1] + ph_3[i0_i1_fused_i2_fused_1]));
  }
}


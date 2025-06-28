void default_function_kernel(float* T_divide, float* T_subtract, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 30; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 13; ++ax2) {
      T_divide[((ax0_ax1_fused * 13) + ax2)] = (ph_0[((ax0_ax1_fused * 13) + ax2)] / ph_3[((ax0_ax1_fused * 13) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 5; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 6; ++ax1) {
      for (int32_t ax2_1 = 0; ax2_1 < 13; ++ax2_1) {
        T_subtract[(((ax0 * 78) + (ax1 * 13)) + ax2_1)] = (ph_0[(((ax0 * 78) + (ax1 * 13)) + ax2_1)] - ph_3[(((ax0 * 78) + (ax1 * 13)) + ax2_1)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 390; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = fabsf(atanhf(ph_0[i0_i1_fused_i2_fused]));
  }
}


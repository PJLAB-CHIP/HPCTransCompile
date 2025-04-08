void default_function_kernel(float* T_multiply, float* compute, float* compute_1, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 20; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 18; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 14; ++ax2) {
        T_multiply[(((ax0 * 252) + (ax1 * 14)) + ax2)] = ((ph_0[(((ax0 * 252) + (ax1 * 14)) + ax2)] * (ph_0[(((ax0 * 252) + (ax1 * 14)) + ax2)] / ph_3[(((ax0 * 252) + (ax1 * 14)) + ax2)])) * ph_0[(((ax0 * 252) + (ax1 * 14)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 5040; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = cosf((ph_0[i0_i1_fused_i2_fused] * (ph_0[i0_i1_fused_i2_fused] / ph_3[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 5040; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = atanhf((ph_0[i0_i1_fused_i2_fused_1] + ph_3[i0_i1_fused_i2_fused_1]));
  }
}


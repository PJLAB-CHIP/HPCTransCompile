void default_function_kernel(float* T_mod, float* compute, float* ph_0) {
  float compute_1[663];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 663; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 39; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 17; ++i2) {
      compute_1[((i0_i1_fused * 17) + i2)] = expf(ph_0[((i0_i1_fused * 17) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 39; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 17; ++ax2) {
      T_mod[((ax0_ax1_fused * 17) + ax2)] = fmodf(ph_0[((ax0_ax1_fused * 17) + ax2)], (ph_0[((ax0_ax1_fused * 17) + ax2)] * atanf(compute_1[((ax0_ax1_fused * 17) + ax2)])));
    }
  }
}


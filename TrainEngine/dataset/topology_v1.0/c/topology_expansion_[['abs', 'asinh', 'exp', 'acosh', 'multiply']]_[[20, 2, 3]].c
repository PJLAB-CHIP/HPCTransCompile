void default_function_kernel(float* T_multiply, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 120; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = fabsf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 40; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 3; ++i2) {
      compute_1[((i0_i1_fused * 3) + i2)] = expf(asinhf(ph_0[((i0_i1_fused * 3) + i2)]));
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 20; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 2; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 3; ++ax2) {
        T_multiply[(((ax0 * 6) + (ax1 * 3)) + ax2)] = (ph_0[(((ax0 * 6) + (ax1 * 3)) + ax2)] * acoshf(ph_0[(((ax0 * 6) + (ax1 * 3)) + ax2)]));
      }
    }
  }
}


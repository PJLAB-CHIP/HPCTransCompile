void default_function_kernel(float* T_multiply, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 84; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 12; ++ax2) {
      T_multiply[((ax0_ax1_fused * 12) + ax2)] = (ph_0[((ax0_ax1_fused * 12) + ax2)] * fabsf(ph_0[((ax0_ax1_fused * 12) + ax2)]));
    }
  }
}


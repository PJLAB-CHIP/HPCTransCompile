void default_function_kernel(float* T_add, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 380; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 11; ++ax2) {
      T_add[((ax0_ax1_fused * 11) + ax2)] = (ph_0[((ax0_ax1_fused * 11) + ax2)] + ceilf(ph_0[((ax0_ax1_fused * 11) + ax2)]));
    }
  }
}


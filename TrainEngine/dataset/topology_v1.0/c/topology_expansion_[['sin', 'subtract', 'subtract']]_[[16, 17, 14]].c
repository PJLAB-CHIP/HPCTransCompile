void default_function_kernel(float* T_subtract, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 272; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 14; ++ax2) {
      T_subtract[((ax0_ax1_fused * 14) + ax2)] = ((ph_0[((ax0_ax1_fused * 14) + ax2)] - sinf(ph_0[((ax0_ax1_fused * 14) + ax2)])) - ph_0[((ax0_ax1_fused * 14) + ax2)]);
    }
  }
}


void default_function_kernel(float* T_sign, float* data) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 70; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 13; ++ax2) {
      T_sign[((ax0_ax1_fused * 13) + ax2)] = ((0.000000e+00f < data[((ax0_ax1_fused * 13) + ax2)]) ? 1.000000e+00f : ((data[((ax0_ax1_fused * 13) + ax2)] < 0.000000e+00f) ? -1.000000e+00f : 0.000000e+00f));
    }
  }
}


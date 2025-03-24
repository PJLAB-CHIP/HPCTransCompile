void default_function_kernel(float* T_transpose, float* ph) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 100; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
      for (int32_t ax3_inner = 0; ax3_inner < 10; ++ax3_inner) {
        T_transpose[(((ax0_ax1_fused * 100) + (ax2 * 10)) + ax3_inner)] = tanhf(max(min(log2f(ph[(((((ax3_inner * 31944) + ((ax0_ax1_fused / 10) * 1452)) + (ax2 * 44)) + (ax0_ax1_fused % 10)) + 16973)]), 1.000000e+01f), 0.000000e+00f));
      }
    }
  }
}


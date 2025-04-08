void default_function_kernel(float* T_divide, float* T_mod, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 20; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 19; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
        T_divide[(((ax0 * 190) + (ax1 * 10)) + ax2)] = (ph_0[(((ax0 * 190) + (ax1 * 10)) + ax2)] / ph_3[(((ax0 * 190) + (ax1 * 10)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 380; ++ax0_ax1_fused) {
    for (int32_t ax2_1 = 0; ax2_1 < 10; ++ax2_1) {
      T_mod[((ax0_ax1_fused * 10) + ax2_1)] = fmodf((ph_0[((ax0_ax1_fused * 10) + ax2_1)] / fmodf(ph_0[((ax0_ax1_fused * 10) + ax2_1)], ph_3[((ax0_ax1_fused * 10) + ax2_1)])), ph_0[((ax0_ax1_fused * 10) + ax2_1)]);
    }
  }
}


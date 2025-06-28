void default_function_kernel(float* T_divide, float* data, float* moving_mean, float* moving_var) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 12; ++ax0_ax1_fused) {
    float T_reshape[1];
    float T_reshape_1[1];
    T_reshape[0] = moving_var[(ax0_ax1_fused & 3)];
    for (int32_t ax2 = 0; ax2 < 2; ++ax2) {
      T_reshape_1[0] = moving_mean[(ax0_ax1_fused & 3)];
      for (int32_t ax3 = 0; ax3 < 10; ++ax3) {
        T_divide[(((ax0_ax1_fused * 20) + (ax2 * 10)) + ax3)] = ((data[(((ax0_ax1_fused * 20) + (ax2 * 10)) + ax3)] - T_reshape_1[0]) / sqrtf((T_reshape[0] + 1.000000e-05f)));
      }
    }
  }
}


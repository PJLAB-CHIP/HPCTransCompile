void default_function_kernel(float* T_divide, float* T_subtract, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 84; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 16; ++i2) {
      compute[((i0_i1_fused * 16) + i2)] = acoshf(ph_0[((i0_i1_fused * 16) + i2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 84; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 16; ++ax2) {
      T_subtract[((ax0_ax1_fused * 16) + ax2)] = (fabsf(ph_0[((ax0_ax1_fused * 16) + ax2)]) - ph_0[((ax0_ax1_fused * 16) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1344; ++ax0_ax1_fused_ax2_fused) {
    T_divide[ax0_ax1_fused_ax2_fused] = (fmodf(ph_0[ax0_ax1_fused_ax2_fused], sinf(ph_0[ax0_ax1_fused_ax2_fused])) / ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


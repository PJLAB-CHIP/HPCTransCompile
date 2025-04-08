void default_function_kernel(float* T_divide, float* ph_0) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 140; ++ax0_ax1_fused_ax2_fused) {
    float compute[1];
    compute[0] = expf(ph_0[ax0_ax1_fused_ax2_fused]);
    T_divide[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] / compute[0]);
  }
}


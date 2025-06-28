void default_function_kernel(float* T_divide, float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 32; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = sinf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 32; ++ax0_ax1_fused_ax2_fused) {
    T_divide[ax0_ax1_fused_ax2_fused] = (cosf(ph_0[ax0_ax1_fused_ax2_fused]) / ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 16; ++i0) {
    for (int32_t i2 = 0; i2 < 2; ++i2) {
      compute_1[((i0 * 2) + i2)] = acoshf((ph_0[((i0 * 2) + i2)] - acoshf(ph_0[((i0 * 2) + i2)])));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 16; ++i0_1) {
    for (int32_t i2_1 = 0; i2_1 < 2; ++i2_1) {
      compute_2[((i0_1 * 2) + i2_1)] = atanf((ph_0[((i0_1 * 2) + i2_1)] - acoshf(ph_0[((i0_1 * 2) + i2_1)])));
    }
  }
}


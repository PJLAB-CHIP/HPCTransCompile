void default_function_kernel(float* T_divide, float* compute, float* compute_1, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 3000; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 200; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 15; ++ax2) {
      T_divide[((ax0_ax1_fused * 15) + ax2)] = (acoshf(ph_0[((ax0_ax1_fused * 15) + ax2)]) / ph_0[((ax0_ax1_fused * 15) + ax2)]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 200; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 15; ++i2) {
      compute_1[((i0_i1_fused * 15) + i2)] = ceilf(fmodf(ph_0[((i0_i1_fused * 15) + i2)], atanf(ph_0[((i0_i1_fused * 15) + i2)])));
    }
  }
}


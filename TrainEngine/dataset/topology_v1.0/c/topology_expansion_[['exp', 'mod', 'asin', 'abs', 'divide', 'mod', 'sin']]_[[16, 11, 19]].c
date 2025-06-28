void default_function_kernel(float* T_mod, float* compute, float* compute_1, float* ph_0) {
  float compute_2[3344];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 3344; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 3344; ++i0_i1_fused_i2_fused_1) {
    compute[i0_i1_fused_i2_fused_1] = asinf(fmodf(ph_0[i0_i1_fused_i2_fused_1], compute_2[i0_i1_fused_i2_fused_1]));
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 3344; ++ax0_ax1_fused_ax2_fused) {
    T_mod[ax0_ax1_fused_ax2_fused] = fmodf((ph_0[ax0_ax1_fused_ax2_fused] / fabsf(ph_0[ax0_ax1_fused_ax2_fused])), ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 176; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 19; ++i2) {
      compute_1[((i0_i1_fused * 19) + i2)] = sinf((ph_0[((i0_i1_fused * 19) + i2)] / fabsf(ph_0[((i0_i1_fused * 19) + i2)])));
    }
  }
}


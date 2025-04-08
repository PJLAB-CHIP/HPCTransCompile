void default_function_kernel(float* T_divide, float* compute, float* compute_1, float* compute_2, float* compute_3, float* ph_0) {
  float compute_4[3536];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 3536; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 3536; ++i0_i1_fused_i2_fused_1) {
    compute_4[i0_i1_fused_i2_fused_1] = expf(ph_0[i0_i1_fused_i2_fused_1]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 3536; ++i0_i1_fused_i2_fused_2) {
    compute_1[i0_i1_fused_i2_fused_2] = sinf(compute_4[i0_i1_fused_i2_fused_2]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_3 = 0; i0_i1_fused_i2_fused_3 < 3536; ++i0_i1_fused_i2_fused_3) {
    compute_2[i0_i1_fused_i2_fused_3] = acoshf(ph_0[i0_i1_fused_i2_fused_3]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_4 = 0; i0_i1_fused_i2_fused_4 < 3536; ++i0_i1_fused_i2_fused_4) {
    compute_3[i0_i1_fused_i2_fused_4] = acoshf(ceilf(ph_0[i0_i1_fused_i2_fused_4]));
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 3536; ++ax0_ax1_fused_ax2_fused) {
    T_divide[ax0_ax1_fused_ax2_fused] = (ceilf(ph_0[ax0_ax1_fused_ax2_fused]) / ph_0[ax0_ax1_fused_ax2_fused]);
  }
}


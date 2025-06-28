void default_function_kernel(float* T_batch_matmul_NN, float* compute, float* compute_1, float* ph_0, float* ph_4, float* ph_6) {
  float auto_scheduler_layout_transform[19];
  for (int32_t ax2 = 0; ax2 < 19; ++ax2) {
    auto_scheduler_layout_transform[ax2] = ph_4[ax2];
  }
  #pragma omp parallel for
  for (int32_t b = 0; b < 4; ++b) {
    T_batch_matmul_NN[b] = 0.000000e+00f;
    for (int32_t k = 0; k < 19; ++k) {
      T_batch_matmul_NN[b] = (T_batch_matmul_NN[b] + (acoshf(ph_0[((b * 19) + k)]) * auto_scheduler_layout_transform[k]));
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 4; ++i0) {
    for (int32_t i2 = 0; i2 < 19; ++i2) {
      compute[((i0 * 19) + i2)] = fabsf((ph_0[((i0 * 19) + i2)] - ph_6[((i0 * 19) + i2)]));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 76; ++i0_i1_fused_i2_fused) {
    compute_1[i0_i1_fused_i2_fused] = atanf(asinhf(ph_0[i0_i1_fused_i2_fused]));
  }
}


void default_function_kernel(float* T_batch_matmul_NN, float* compute, float* ph_0, float* ph_4) {
  float auto_scheduler_layout_transform[10];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 170; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = expf(ph_0[i0_i1_fused_i2_fused]);
  }
  for (int32_t ax4 = 0; ax4 < 2; ++ax4) {
    for (int32_t ax8 = 0; ax8 < 5; ++ax8) {
      auto_scheduler_layout_transform[((ax4 * 5) + ax8)] = ph_4[((ax8 * 2) + ax4)];
    }
  }
  for (int32_t b_inner_init = 0; b_inner_init < 5; ++b_inner_init) {
    for (int32_t i_inner_init = 0; i_inner_init < 17; ++i_inner_init) {
      T_batch_matmul_NN[((b_inner_init * 17) + i_inner_init)] = 0.000000e+00f;
    }
  }
  for (int32_t k_outer = 0; k_outer < 2; ++k_outer) {
    for (int32_t b_inner = 0; b_inner < 5; ++b_inner) {
      for (int32_t i_inner = 0; i_inner < 17; ++i_inner) {
        T_batch_matmul_NN[((b_inner * 17) + i_inner)] = (T_batch_matmul_NN[((b_inner * 17) + i_inner)] + (fabsf(ph_0[(((b_inner * 34) + (i_inner * 2)) + k_outer)]) * auto_scheduler_layout_transform[((k_outer * 5) + b_inner)]));
      }
    }
  }
}


void default_function_kernel(float* T_subtract, float* compute, float* compute_1, float* compute_2, float* ph_0, float* ph_3) {
  float auto_scheduler_layout_transform[120];
  float T_batch_matmul_NN[20];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 24; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = cosf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 24; ++ax0_ax1_fused_ax2_fused) {
    T_subtract[ax0_ax1_fused_ax2_fused] = (asinhf(ph_0[ax0_ax1_fused_ax2_fused]) - ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 24; ++i0_i1_fused_i2_fused_1) {
    compute_1[i0_i1_fused_i2_fused_1] = asinhf(asinhf(ph_0[i0_i1_fused_i2_fused_1]));
  }
  for (int32_t ax4 = 0; ax4 < 3; ++ax4) {
    for (int32_t ax5 = 0; ax5 < 4; ++ax5) {
      for (int32_t ax6 = 0; ax6 < 5; ++ax6) {
        for (int32_t ax7 = 0; ax7 < 2; ++ax7) {
          auto_scheduler_layout_transform[((((ax4 * 40) + (ax5 * 10)) + (ax6 * 2)) + ax7)] = ph_3[((((ax5 * 30) + (ax4 * 10)) + (ax7 * 5)) + ax6)];
        }
      }
    }
  }
  for (int32_t b_outer_inner_init = 0; b_outer_inner_init < 4; ++b_outer_inner_init) {
    for (int32_t j_outer_inner_init = 0; j_outer_inner_init < 5; ++j_outer_inner_init) {
      T_batch_matmul_NN[((b_outer_inner_init * 5) + j_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int32_t k_outer = 0; k_outer < 3; ++k_outer) {
    for (int32_t b_outer_inner = 0; b_outer_inner < 4; ++b_outer_inner) {
      for (int32_t j_outer_inner = 0; j_outer_inner < 5; ++j_outer_inner) {
        for (int32_t k_inner = 0; k_inner < 2; ++k_inner) {
          T_batch_matmul_NN[((b_outer_inner * 5) + j_outer_inner)] = (T_batch_matmul_NN[((b_outer_inner * 5) + j_outer_inner)] + (ph_0[(((b_outer_inner * 6) + (k_outer * 2)) + k_inner)] * auto_scheduler_layout_transform[((((k_outer * 40) + (b_outer_inner * 10)) + (j_outer_inner * 2)) + k_inner)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_2 = 0; i0_i1_fused_i2_fused_2 < 20; ++i0_i1_fused_i2_fused_2) {
    compute_2[i0_i1_fused_i2_fused_2] = expf(T_batch_matmul_NN[i0_i1_fused_i2_fused_2]);
  }
}


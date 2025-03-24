void default_function_kernel(float* T_batch_matmul_NN, float* compute, float* ph_0) {
  float auto_scheduler_layout_transform[1280];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1280; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = asinhf(fmodf(ph_0[i0_i1_fused_i2_fused], ceilf(ph_0[i0_i1_fused_i2_fused])));
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 5; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
      for (int32_t ax5 = 0; ax5 < 2; ++ax5) {
        for (int32_t ax6 = 0; ax6 < 8; ++ax6) {
          for (int32_t ax7 = 0; ax7 < 2; ++ax7) {
            for (int32_t ax8 = 0; ax8 < 2; ++ax8) {
              auto_scheduler_layout_transform[((((((ax0_ax1_fused_ax2_fused * 256) + (ax4 * 64)) + (ax5 * 32)) + (ax6 * 4)) + (ax7 * 2)) + ax8)] = atanf(ph_0[((((((ax0_ax1_fused_ax2_fused * 256) + (ax5 * 128)) + (ax8 * 64)) + (ax4 * 16)) + (ax7 * 8)) + ax6)]);
            }
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused = 0; b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused < 10; ++b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused) {
    for (int32_t b_outer_inner_init = 0; b_outer_inner_init < 2; ++b_outer_inner_init) {
      for (int32_t i_outer_inner_init = 0; i_outer_inner_init < 2; ++i_outer_inner_init) {
        for (int32_t j_outer_inner_init = 0; j_outer_inner_init < 8; ++j_outer_inner_init) {
          for (int32_t b_inner_init = 0; b_inner_init < 2; ++b_inner_init) {
            for (int32_t i_inner_init = 0; i_inner_init < 2; ++i_inner_init) {
              T_batch_matmul_NN[((((((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused % 5) * 256) + (b_outer_inner_init * 128)) + (b_inner_init * 64)) + ((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused / 5) * 32)) + (i_outer_inner_init * 16)) + (i_inner_init * 8)) + j_outer_inner_init)] = 0.000000e+00f;
            }
          }
        }
      }
    }
    for (int32_t k_outer = 0; k_outer < 4; ++k_outer) {
      for (int32_t b_outer_inner = 0; b_outer_inner < 2; ++b_outer_inner) {
        for (int32_t i_outer_inner = 0; i_outer_inner < 2; ++i_outer_inner) {
          for (int32_t j_outer_inner = 0; j_outer_inner < 8; ++j_outer_inner) {
            for (int32_t k_inner = 0; k_inner < 2; ++k_inner) {
              for (int32_t b_inner = 0; b_inner < 2; ++b_inner) {
                for (int32_t i_inner = 0; i_inner < 2; ++i_inner) {
                  T_batch_matmul_NN[((((((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused % 5) * 256) + (b_outer_inner * 128)) + (b_inner * 64)) + ((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused / 5) * 32)) + (i_outer_inner * 16)) + (i_inner * 8)) + j_outer_inner)] = (T_batch_matmul_NN[((((((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused % 5) * 256) + (b_outer_inner * 128)) + (b_inner * 64)) + ((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused / 5) * 32)) + (i_outer_inner * 16)) + (i_inner * 8)) + j_outer_inner)] + (ph_0[(((((((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused % 5) * 256) + (b_outer_inner * 128)) + (b_inner * 64)) + ((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused / 5) * 32)) + (i_outer_inner * 16)) + (i_inner * 8)) + (k_outer * 2)) + k_inner)] * auto_scheduler_layout_transform[(((((((b_outer_outer_outer_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused % 5) * 256) + (k_outer * 64)) + (b_outer_inner * 32)) + (j_outer_inner * 4)) + (k_inner * 2)) + b_inner)]));
                }
              }
            }
          }
        }
      }
    }
  }
}


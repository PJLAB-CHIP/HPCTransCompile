void default_function_kernel(float* T_add, float* compute, float* compute_1, float* ph_0, float* ph_6) {
  float auto_scheduler_layout_transform[1440];
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 3456; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = acoshf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 3456; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = (ph_0[ax0_ax1_fused_ax2_fused] + (atanhf(ph_0[ax0_ax1_fused_ax2_fused]) / ph_0[ax0_ax1_fused_ax2_fused]));
  }
  for (int32_t ax3 = 0; ax3 < 5; ++ax3) {
    for (int32_t ax8 = 0; ax8 < 9; ++ax8) {
      for (int32_t ax9 = 0; ax9 < 4; ++ax9) {
        for (int32_t ax11 = 0; ax11 < 2; ++ax11) {
          for (int32_t ax12 = 0; ax12 < 4; ++ax12) {
            auto_scheduler_layout_transform[(((((ax3 * 288) + (ax8 * 32)) + (ax9 * 8)) + (ax11 * 4)) + ax12)] = ph_6[(((((ax9 * 360) + (ax12 * 90)) + (ax8 * 10)) + (ax11 * 5)) + ax3)];
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused = 0; i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused < 10; ++i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused) {
    float T_batch_matmul_NN[96];
    for (int32_t b_outer_inner_init = 0; b_outer_inner_init < 4; ++b_outer_inner_init) {
      for (int32_t i_outer_inner_init = 0; i_outer_inner_init < 3; ++i_outer_inner_init) {
        for (int32_t b_inner_init = 0; b_inner_init < 4; ++b_inner_init) {
          for (int32_t i_inner_init = 0; i_inner_init < 2; ++i_inner_init) {
            T_batch_matmul_NN[((((b_outer_inner_init * 24) + (b_inner_init * 6)) + (i_outer_inner_init * 2)) + i_inner_init)] = 0.000000e+00f;
          }
        }
      }
    }
    for (int32_t k_outer = 0; k_outer < 9; ++k_outer) {
      for (int32_t b_outer_inner = 0; b_outer_inner < 4; ++b_outer_inner) {
        for (int32_t i_outer_inner = 0; i_outer_inner < 3; ++i_outer_inner) {
          for (int32_t k_inner = 0; k_inner < 2; ++k_inner) {
            for (int32_t b_inner = 0; b_inner < 4; ++b_inner) {
              for (int32_t i_inner = 0; i_inner < 2; ++i_inner) {
                T_batch_matmul_NN[((((b_outer_inner * 24) + (b_inner * 6)) + (i_outer_inner * 2)) + i_inner)] = (T_batch_matmul_NN[((((b_outer_inner * 24) + (b_inner * 6)) + (i_outer_inner * 2)) + i_inner)] + (ph_0[(((((((b_outer_inner * 864) + (b_inner * 216)) + ((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused / 5) * 108)) + (i_outer_inner * 36)) + (i_inner * 18)) + (k_outer * 2)) + k_inner)] * auto_scheduler_layout_transform[((((((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused % 5) * 288) + (k_outer * 32)) + (b_outer_inner * 8)) + (k_inner * 4)) + b_inner)]));
              }
            }
          }
        }
      }
    }
    for (int32_t i0_inner = 0; i0_inner < 16; ++i0_inner) {
      for (int32_t i1_inner = 0; i1_inner < 6; ++i1_inner) {
        compute_1[((((i0_inner * 60) + ((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused / 5) * 30)) + (i1_inner * 5)) + (i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused % 5))] = fabsf(T_batch_matmul_NN[((i0_inner * 6) + i1_inner)]);
      }
    }
  }
}


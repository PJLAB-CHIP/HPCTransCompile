void default_function_kernel(float* T_add, float* compute, float* compute_1, float* compute_2, float* ph_0, float* ph_6) {
  float auto_scheduler_layout_transform[640];
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 512; ++ax0_ax1_fused_ax2_fused) {
    T_add[ax0_ax1_fused_ax2_fused] = (ceilf(ph_0[ax0_ax1_fused_ax2_fused]) + ph_0[ax0_ax1_fused_ax2_fused]);
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 512; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = sinf(ph_0[i0_i1_fused_i2_fused]);
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_ax2_fused_1 = 0; ax0_ax1_fused_ax2_fused_1 < 16; ++ax0_ax1_fused_ax2_fused_1) {
    for (int32_t ax3 = 0; ax3 < 5; ++ax3) {
      for (int32_t ax11 = 0; ax11 < 8; ++ax11) {
        auto_scheduler_layout_transform[(((ax0_ax1_fused_ax2_fused_1 * 40) + (ax3 * 8)) + ax11)] = ph_6[(((ax0_ax1_fused_ax2_fused_1 * 40) + (ax11 * 5)) + ax3)];
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused = 0; i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused < 160; ++i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused) {
    float T_batch_matmul_NN[2];
    for (int32_t i_inner_init = 0; i_inner_init < 2; ++i_inner_init) {
      T_batch_matmul_NN[i_inner_init] = 0.000000e+00f;
    }
    for (int32_t k_inner = 0; k_inner < 8; ++k_inner) {
      for (int32_t i_inner = 0; i_inner < 2; ++i_inner) {
        T_batch_matmul_NN[i_inner] = (T_batch_matmul_NN[i_inner] + (ph_0[((((((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused / 40) * 128) + (((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused % 20) / 5) * 32)) + (((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused % 40) / 20) * 16)) + (i_inner * 8)) + k_inner)] * auto_scheduler_layout_transform[((((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused / 40) * 160) + ((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused % 20) * 8)) + k_inner)]));
      }
    }
    for (int32_t i1_inner = 0; i1_inner < 2; ++i1_inner) {
      compute_1[((((((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused / 40) * 80) + (((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused % 20) / 5) * 20)) + (((i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused % 40) / 20) * 10)) + (i1_inner * 5)) + (i0_outer_outer_i1_outer_outer_fused_i2_outer_outer_fused_i0_outer_inner_fused_i1_outer_inner_fused_i2_outer_inner_fused % 5))] = asinhf(T_batch_matmul_NN[i1_inner]);
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused_1 = 0; i0_i1_fused_i2_fused_1 < 512; ++i0_i1_fused_i2_fused_1) {
    compute_2[i0_i1_fused_i2_fused_1] = fabsf(sinf(ph_0[i0_i1_fused_i2_fused_1]));
  }
}


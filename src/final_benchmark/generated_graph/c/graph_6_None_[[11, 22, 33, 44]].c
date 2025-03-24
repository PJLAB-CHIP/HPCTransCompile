void default_function_kernel(float* T_transpose, float* ph) {
  float T_softmax_maxelem[1800];
  float compute[1800];
  for (int32_t i0 = 0; i0 < 20; ++i0) {
    for (int32_t i1 = 0; i1 < 90; ++i1) {
      T_softmax_maxelem[((i0 * 90) + i1)] = -3.402823e+38f;
      for (int32_t k = 0; k < 10; ++k) {
        T_softmax_maxelem[((i0 * 90) + i1)] = max(T_softmax_maxelem[((i0 * 90) + i1)], ((10 <= i0) ? ph[(((((i0 * 31944) + (k * 1452)) + ((i1 / 9) * 44)) + (i1 % 9)) - 269026)] : ph[(((((i0 * 31944) + (k * 1452)) + ((i1 / 9) * 44)) + (i1 % 9)) + 16973)]));
      }
    }
  }
  for (int32_t i0_1 = 0; i0_1 < 20; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 90; ++i1_1) {
      compute[((i0_1 * 90) + i1_1)] = 0.000000e+00f;
      for (int32_t k_1 = 0; k_1 < 10; ++k_1) {
        compute[((i0_1 * 90) + i1_1)] = (compute[((i0_1 * 90) + i1_1)] + expf((((10 <= i0_1) ? ph[(((((i0_1 * 31944) + (k_1 * 1452)) + ((i1_1 / 9) * 44)) + (i1_1 % 9)) - 269026)] : ph[(((((i0_1 * 31944) + (k_1 * 1452)) + ((i1_1 / 9) * 44)) + (i1_1 % 9)) + 16973)]) - T_softmax_maxelem[((i0_1 * 90) + i1_1)])));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 1800; ++ax0_ax1_fused) {
    for (int32_t ax2_inner = 0; ax2_inner < 10; ++ax2_inner) {
      T_transpose[((ax0_ax1_fused * 10) + ax2_inner)] = ((((900 <= ax0_ax1_fused) ? ph[((((((ax0_ax1_fused / 90) * 31944) + (ax2_inner * 1452)) + (((ax0_ax1_fused % 90) / 9) * 44)) + (ax0_ax1_fused % 9)) - 269026)] : ph[((((((ax0_ax1_fused / 90) * 31944) + (ax2_inner * 1452)) + (((ax0_ax1_fused % 90) / 9) * 44)) + (ax0_ax1_fused % 9)) + 16973)]) - T_softmax_maxelem[ax0_ax1_fused]) - logf(compute[ax0_ax1_fused]));
    }
  }
}


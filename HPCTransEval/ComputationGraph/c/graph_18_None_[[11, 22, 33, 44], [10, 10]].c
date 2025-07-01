void default_function_kernel(float* T_cast, float* ph, float* ph_1) {
  float T_softmax_maxelem[2000];
  float T_softmax_expsum[2000];
  float T_multiply_red[180];
  for (int32_t i0 = 0; i0 < 20; ++i0) {
    for (int32_t i1 = 0; i1 < 10; ++i1) {
      for (int32_t i2 = 0; i2 < 10; ++i2) {
        T_softmax_maxelem[(((i0 * 100) + (i1 * 10)) + i2)] = -3.402823e+38f;
        for (int32_t k = 0; k < 9; ++k) {
          T_softmax_maxelem[(((i0 * 100) + (i1 * 10)) + i2)] = max(T_softmax_maxelem[(((i0 * 100) + (i1 * 10)) + i2)], ((10 <= i0) ? ph[(((((i0 * 31944) + (i1 * 1452)) + (i2 * 44)) + k) - 269026)] : ph[(((((i0 * 31944) + (i1 * 1452)) + (i2 * 44)) + k) + 16973)]));
        }
      }
    }
  }
  for (int32_t i0_1 = 0; i0_1 < 20; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 10; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 10; ++i2_1) {
        T_softmax_expsum[(((i0_1 * 100) + (i1_1 * 10)) + i2_1)] = 0.000000e+00f;
        for (int32_t k_1 = 0; k_1 < 9; ++k_1) {
          T_softmax_expsum[(((i0_1 * 100) + (i1_1 * 10)) + i2_1)] = (T_softmax_expsum[(((i0_1 * 100) + (i1_1 * 10)) + i2_1)] + expf((((10 <= i0_1) ? ph[(((((i0_1 * 31944) + (i1_1 * 1452)) + (i2_1 * 44)) + k_1) - 269026)] : ph[(((((i0_1 * 31944) + (i1_1 * 1452)) + (i2_1 * 44)) + k_1) + 16973)]) - T_softmax_maxelem[(((i0_1 * 100) + (i1_1 * 10)) + i2_1)])));
        }
      }
    }
  }
  for (int32_t ax0 = 0; ax0 < 20; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 9; ++ax1) {
      T_multiply_red[((ax0 * 9) + ax1)] = 0.000000e+00f;
      for (int32_t k1 = 0; k1 < 10; ++k1) {
        for (int32_t k2 = 0; k2 < 10; ++k2) {
          int cse_var_9 = (10 <= ax0);
          T_multiply_red[((ax0 * 9) + ax1)] = (T_multiply_red[((ax0 * 9) + ax1)] + ((expf(((cse_var_9 ? ph[(((((ax0 * 31944) + (k2 * 1452)) + (k1 * 44)) + ax1) - 269026)] : ph[(((((ax0 * 31944) + (k2 * 1452)) + (k1 * 44)) + ax1) + 16973)]) - T_softmax_maxelem[(((ax0 * 100) + (k2 * 10)) + k1)])) / T_softmax_expsum[(((ax0 * 100) + (k2 * 10)) + k1)]) * (expf(((cse_var_9 ? ph[(((((ax0 * 31944) + (k2 * 1452)) + (k1 * 44)) + ax1) - 269026)] : ph[(((((ax0 * 31944) + (k2 * 1452)) + (k1 * 44)) + ax1) + 16973)]) - T_softmax_maxelem[(((ax0 * 100) + (k2 * 10)) + k1)])) / T_softmax_expsum[(((ax0 * 100) + (k2 * 10)) + k1)])));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 200; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
      for (int32_t ax3_inner = 0; ax3_inner < 9; ++ax3_inner) {
        T_cast[(((ax0_ax1_fused * 90) + (ax2 * 9)) + ax3_inner)] = (((expf((((100 <= ax0_ax1_fused) ? ph[((((((ax0_ax1_fused / 10) * 31944) + (ax2 * 1452)) + ((ax0_ax1_fused % 10) * 44)) + ax3_inner) - 269026)] : ph[((((((ax0_ax1_fused / 10) * 31944) + (ax2 * 1452)) + ((ax0_ax1_fused % 10) * 44)) + ax3_inner) + 16973)]) - T_softmax_maxelem[((((ax0_ax1_fused / 10) * 100) + (ax2 * 10)) + (ax0_ax1_fused % 10))])) / T_softmax_expsum[((((ax0_ax1_fused / 10) * 100) + (ax2 * 10)) + (ax0_ax1_fused % 10))]) * ph_1[(((ax0_ax1_fused % 10) * 10) + ax2)]) * (1.000000e+00f / sqrtf(((T_multiply_red[(((ax0_ax1_fused / 10) * 9) + ax3_inner)] * 1.000000e-02f) + 1.000000e-05f))));
      }
    }
  }
}


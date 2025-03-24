void default_function_kernel(float* T_softmax_norm, float* ph) {
  float T_softmax_maxelem[8];
  float T_softmax_expsum[8];
  for (int32_t i0 = 0; i0 < 8; ++i0) {
    T_softmax_maxelem[i0] = -3.402823e+38f;
    for (int32_t k = 0; k < 21; ++k) {
      T_softmax_maxelem[i0] = max(T_softmax_maxelem[i0], ph[((((i0 * 60) + (k % 7)) + 109) - ((k / 7) * 12))]);
    }
  }
  for (int32_t i0_1 = 0; i0_1 < 8; ++i0_1) {
    T_softmax_expsum[i0_1] = 0.000000e+00f;
    for (int32_t k_1 = 0; k_1 < 21; ++k_1) {
      T_softmax_expsum[i0_1] = (T_softmax_expsum[i0_1] + expf((ph[((((i0_1 * 60) + (k_1 % 7)) + 109) - ((k_1 / 7) * 12))] - T_softmax_maxelem[i0_1])));
    }
  }
  #pragma omp parallel for
  for (int32_t i0_2 = 0; i0_2 < 8; ++i0_2) {
    for (int32_t i1_outer = 0; i1_outer < 2; ++i1_outer) {
      for (int32_t i1_inner = 0; i1_inner < 16; ++i1_inner) {
        if (((i1_outer * 16) + i1_inner) < 21) {
          T_softmax_norm[(((i0_2 * 21) + (i1_outer * 16)) + i1_inner)] = (expf((ph[((((i0_2 * 60) + (((i1_outer * 2) + i1_inner) % 7)) + 109) - ((((i1_outer * 16) + i1_inner) / 7) * 12))] - T_softmax_maxelem[i0_2])) / T_softmax_expsum[i0_2]);
        }
      }
    }
  }
}


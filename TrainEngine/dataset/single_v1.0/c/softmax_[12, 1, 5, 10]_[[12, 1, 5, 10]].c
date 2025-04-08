void default_function_kernel(float* T_softmax_norm, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 12; ++i0_i1_fused) {
    float T_softmax_exp[10];
    float T_softmax_maxelem[1];
    float T_softmax_expsum[1];
    for (int32_t i2 = 0; i2 < 5; ++i2) {
      for (int32_t i3 = 0; i3 < 10; ++i3) {
        for (int32_t i3_1 = 0; i3_1 < 10; ++i3_1) {
          T_softmax_maxelem[0] = -3.402823e+38f;
          for (int32_t k = 0; k < 10; ++k) {
            T_softmax_maxelem[0] = max(T_softmax_maxelem[0], data[(((i0_i1_fused * 50) + (i2 * 10)) + k)]);
          }
          T_softmax_exp[i3_1] = expf((data[(((i0_i1_fused * 50) + (i2 * 10)) + i3_1)] - T_softmax_maxelem[0]));
        }
        T_softmax_expsum[0] = 0.000000e+00f;
        for (int32_t k_1 = 0; k_1 < 10; ++k_1) {
          T_softmax_expsum[0] = (T_softmax_expsum[0] + T_softmax_exp[k_1]);
        }
        T_softmax_norm[(((i0_i1_fused * 50) + (i2 * 10)) + i3)] = (T_softmax_exp[i3] / T_softmax_expsum[0]);
      }
    }
  }
}


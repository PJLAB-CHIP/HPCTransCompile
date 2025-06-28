void default_function_kernel(float* T_softmax_norm, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 2400; ++i0_i1_fused_i2_fused) {
    float T_softmax_exp[16];
    float T_softmax_maxelem[1];
    float T_softmax_expsum[1];
    for (int32_t i3 = 0; i3 < 16; ++i3) {
      T_softmax_maxelem[0] = -3.402823e+38f;
      for (int32_t k = 0; k < 16; ++k) {
        T_softmax_maxelem[0] = max(T_softmax_maxelem[0], data[((i0_i1_fused_i2_fused * 16) + k)]);
      }
      T_softmax_exp[i3] = expf((data[((i0_i1_fused_i2_fused * 16) + i3)] - T_softmax_maxelem[0]));
    }
    for (int32_t i3_1 = 0; i3_1 < 16; ++i3_1) {
      T_softmax_expsum[0] = 0.000000e+00f;
      for (int32_t k_1 = 0; k_1 < 16; ++k_1) {
        T_softmax_expsum[0] = (T_softmax_expsum[0] + T_softmax_exp[k_1]);
      }
      T_softmax_norm[((i0_i1_fused_i2_fused * 16) + i3_1)] = (T_softmax_exp[i3_1] / T_softmax_expsum[0]);
    }
  }
}


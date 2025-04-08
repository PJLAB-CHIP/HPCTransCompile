void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused = 0; i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused < 1190; ++i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused) {
    float T_softmax_maxelem[2];
    float compute_1[2];
    for (int32_t i1 = 0; i1 < 2; ++i1) {
      T_softmax_maxelem[i1] = -3.402823e+38f;
      for (int32_t k = 0; k < 17; ++k) {
        T_softmax_maxelem[i1] = max(T_softmax_maxelem[i1], data[(((((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused / 238) * 476) + (i1 * 238)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused % 14) * 17)) + k)]);
      }
    }
    for (int32_t i1_1 = 0; i1_1 < 2; ++i1_1) {
      compute_1[i1_1] = 0.000000e+00f;
      for (int32_t k_1 = 0; k_1 < 17; ++k_1) {
        compute_1[i1_1] = (compute_1[i1_1] + expf((data[(((((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused / 238) * 476) + (i1_1 * 238)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused % 14) * 17)) + k_1)] - T_softmax_maxelem[i1_1])));
      }
    }
    for (int32_t i1_inner = 0; i1_inner < 2; ++i1_inner) {
      compute[(((((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused / 238) * 476) + (i1_inner * 238)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused % 14) * 17)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused % 238) / 14))] = ((data[(((((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused / 238) * 476) + (i1_inner * 238)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused % 14) * 17)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused % 238) / 14))] - T_softmax_maxelem[i1_inner]) - logf(compute_1[i1_inner]));
    }
  }
}


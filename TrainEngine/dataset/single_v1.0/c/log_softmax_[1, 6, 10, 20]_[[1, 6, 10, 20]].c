void default_function_kernel(float* compute, float* data) {
  float T_softmax_maxelem[60];
  float compute_1[10];
  for (int32_t i1 = 0; i1 < 6; ++i1) {
    for (int32_t i2 = 0; i2 < 10; ++i2) {
      T_softmax_maxelem[((i1 * 10) + i2)] = -3.402823e+38f;
      for (int32_t k = 0; k < 20; ++k) {
        T_softmax_maxelem[((i1 * 10) + i2)] = max(T_softmax_maxelem[((i1 * 10) + i2)], data[(((i1 * 200) + (i2 * 20)) + k)]);
      }
    }
  }
  for (int32_t i1_outer_outer_inner = 0; i1_outer_outer_inner < 6; ++i1_outer_outer_inner) {
    for (int32_t i2_1 = 0; i2_1 < 10; ++i2_1) {
      compute_1[i2_1] = 0.000000e+00f;
      for (int32_t k_1 = 0; k_1 < 20; ++k_1) {
        compute_1[i2_1] = (compute_1[i2_1] + expf((data[(((i1_outer_outer_inner * 200) + (i2_1 * 20)) + k_1)] - T_softmax_maxelem[((i1_outer_outer_inner * 10) + i2_1)])));
      }
    }
    for (int32_t i3_outer_inner = 0; i3_outer_inner < 20; ++i3_outer_inner) {
      for (int32_t i2_inner = 0; i2_inner < 10; ++i2_inner) {
        compute[(((i1_outer_outer_inner * 200) + (i2_inner * 20)) + i3_outer_inner)] = ((data[(((i1_outer_outer_inner * 200) + (i2_inner * 20)) + i3_outer_inner)] - T_softmax_maxelem[((i1_outer_outer_inner * 10) + i2_inner)]) - logf(compute_1[i2_inner]));
      }
    }
  }
}


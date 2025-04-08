void default_function_kernel(float* compute, float* data) {
  float T_softmax_maxelem[12];
  float compute_1[1];
  for (int32_t i0 = 0; i0 < 4; ++i0) {
    for (int32_t i2 = 0; i2 < 3; ++i2) {
      T_softmax_maxelem[((i0 * 3) + i2)] = -3.402823e+38f;
      for (int32_t k = 0; k < 18; ++k) {
        T_softmax_maxelem[((i0 * 3) + i2)] = max(T_softmax_maxelem[((i0 * 3) + i2)], data[(((i0 * 54) + (i2 * 18)) + k)]);
      }
    }
  }
  for (int32_t i3_outer_outer_outer = 0; i3_outer_outer_outer < 9; ++i3_outer_outer_outer) {
    for (int32_t i0_outer_outer_inner = 0; i0_outer_outer_inner < 2; ++i0_outer_outer_inner) {
      for (int32_t i2_outer_outer_inner = 0; i2_outer_outer_inner < 3; ++i2_outer_outer_inner) {
        for (int32_t i3_outer_outer_inner = 0; i3_outer_outer_inner < 2; ++i3_outer_outer_inner) {
          for (int32_t i0_outer_inner = 0; i0_outer_inner < 2; ++i0_outer_inner) {
            compute_1[0] = 0.000000e+00f;
            for (int32_t k_1 = 0; k_1 < 18; ++k_1) {
              compute_1[0] = (compute_1[0] + expf((data[((((i0_outer_outer_inner * 108) + (i0_outer_inner * 54)) + (i2_outer_outer_inner * 18)) + k_1)] - T_softmax_maxelem[(((i0_outer_outer_inner * 6) + (i0_outer_inner * 3)) + i2_outer_outer_inner)])));
            }
            compute[(((((i0_outer_outer_inner * 108) + (i0_outer_inner * 54)) + (i2_outer_outer_inner * 18)) + (i3_outer_outer_outer * 2)) + i3_outer_outer_inner)] = ((data[(((((i0_outer_outer_inner * 108) + (i0_outer_inner * 54)) + (i2_outer_outer_inner * 18)) + (i3_outer_outer_outer * 2)) + i3_outer_outer_inner)] - T_softmax_maxelem[(((i0_outer_outer_inner * 6) + (i0_outer_inner * 3)) + i2_outer_outer_inner)]) - logf(compute_1[0]));
          }
        }
      }
    }
  }
}


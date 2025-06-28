void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused = 0; i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused < 156; ++i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused) {
    float T_softmax_maxelem[27];
    float compute_1[27];
    for (int32_t i0 = 0; i0 < 9; ++i0) {
      for (int32_t i2 = 0; i2 < 3; ++i2) {
        T_softmax_maxelem[((i0 * 3) + i2)] = -3.402823e+38f;
        for (int32_t k = 0; k < 15; ++k) {
          T_softmax_maxelem[((i0 * 3) + i2)] = max(T_softmax_maxelem[((i0 * 3) + i2)], data[((((((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused & 1) * 31590) + (i0 * 3510)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused >> 1) * 45)) + (i2 * 15)) + k)]);
        }
      }
    }
    for (int32_t i0_1 = 0; i0_1 < 9; ++i0_1) {
      for (int32_t i2_1 = 0; i2_1 < 3; ++i2_1) {
        compute_1[((i0_1 * 3) + i2_1)] = 0.000000e+00f;
        for (int32_t k_1 = 0; k_1 < 15; ++k_1) {
          compute_1[((i0_1 * 3) + i2_1)] = (compute_1[((i0_1 * 3) + i2_1)] + expf((data[((((((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused & 1) * 31590) + (i0_1 * 3510)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused >> 1) * 45)) + (i2_1 * 15)) + k_1)] - T_softmax_maxelem[((i0_1 * 3) + i2_1)])));
        }
      }
    }
    for (int32_t i3_outer_outer_inner = 0; i3_outer_outer_inner < 3; ++i3_outer_outer_inner) {
      for (int32_t i0_outer_inner = 0; i0_outer_inner < 9; ++i0_outer_inner) {
        for (int32_t i2_outer_inner = 0; i2_outer_inner < 3; ++i2_outer_inner) {
          for (int32_t i3_outer_inner = 0; i3_outer_inner < 5; ++i3_outer_inner) {
            compute[(((((((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused & 1) * 31590) + (i0_outer_inner * 3510)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused >> 1) * 45)) + (i2_outer_inner * 15)) + (i3_outer_outer_inner * 5)) + i3_outer_inner)] = ((data[(((((((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused & 1) * 31590) + (i0_outer_inner * 3510)) + ((i0_outer_outer_outer_i1_outer_outer_outer_fused_i2_outer_outer_outer_fused_i3_outer_outer_outer_fused_i0_outer_outer_inner_fused_i1_outer_outer_inner_fused >> 1) * 45)) + (i2_outer_inner * 15)) + (i3_outer_outer_inner * 5)) + i3_outer_inner)] - T_softmax_maxelem[((i0_outer_inner * 3) + i2_outer_inner)]) - logf(compute_1[((i0_outer_inner * 3) + i2_outer_inner)]));
          }
        }
      }
    }
  }
}


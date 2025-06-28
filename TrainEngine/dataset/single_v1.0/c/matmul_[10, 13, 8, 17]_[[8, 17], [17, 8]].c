void default_function_kernel(float* T_matmul, float* left_matrix, float* right_matrix) {
  for (int32_t ax1_outer_inner_init = 0; ax1_outer_inner_init < 8; ++ax1_outer_inner_init) {
    for (int32_t ax0_inner_init = 0; ax0_inner_init < 8; ++ax0_inner_init) {
      T_matmul[((ax0_inner_init * 8) + ax1_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int32_t k_outer = 0; k_outer < 17; ++k_outer) {
    for (int32_t ax1_outer_inner = 0; ax1_outer_inner < 8; ++ax1_outer_inner) {
      for (int32_t ax0_inner = 0; ax0_inner < 8; ++ax0_inner) {
        T_matmul[((ax0_inner * 8) + ax1_outer_inner)] = (T_matmul[((ax0_inner * 8) + ax1_outer_inner)] + (left_matrix[((ax0_inner * 17) + k_outer)] * right_matrix[((k_outer * 8) + ax1_outer_inner)]));
      }
    }
  }
}


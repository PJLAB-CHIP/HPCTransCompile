void default_function_kernel(float* Scale, float* ScaleShift, float* Shift, float* data) {
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_c_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused = 0; b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_c_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused < 57; ++b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_c_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused) {
    for (int32_t c_outer_inner = 0; c_outer_inner < 5; ++c_outer_inner) {
      for (int32_t i_outer_inner = 0; i_outer_inner < 20; ++i_outer_inner) {
        for (int32_t b_inner = 0; b_inner < 5; ++b_inner) {
          ScaleShift[((((((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_c_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused / 19) * 9500) + (b_inner * 1900)) + (c_outer_inner * 380)) + (i_outer_inner * 19)) + (b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_c_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused % 19))] = ((data[((((((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_c_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused / 19) * 9500) + (b_inner * 1900)) + (c_outer_inner * 380)) + (i_outer_inner * 19)) + (b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_b_outer_outer_inner_fused_c_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused % 19))] * Scale[c_outer_inner]) + Shift[c_outer_inner]);
        }
      }
    }
  }
}


void default_function_kernel(float* Scale, float* ScaleShift, float* Shift, float* data) {
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused = 0; b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused < 1235; ++b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused) {
    for (int32_t i_outer_outer_inner = 0; i_outer_outer_inner < 7; ++i_outer_outer_inner) {
      for (int32_t i_outer_inner = 0; i_outer_inner < 2; ++i_outer_inner) {
        ScaleShift[(((((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused / 19) * 266) + (i_outer_outer_inner * 38)) + (i_outer_inner * 19)) + (b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused % 19))] = ((data[(((((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused / 19) * 266) + (i_outer_outer_inner * 38)) + (i_outer_inner * 19)) + (b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused % 19))] * Scale[((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused % 95) / 19)]) + Shift[((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused % 95) / 19)]);
      }
    }
  }
}


void default_function_kernel(float* Scale, float* ScaleShift, float* Shift, float* data) {
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused = 0; b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused < 120; ++b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused) {
    for (int32_t i_outer_outer_inner = 0; i_outer_outer_inner < 3; ++i_outer_outer_inner) {
      for (int32_t b_outer_inner = 0; b_outer_inner < 13; ++b_outer_inner) {
        for (int32_t c_inner = 0; c_inner < 3; ++c_inner) {
          ScaleShift[((((((b_outer_inner * 1080) + ((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused / 24) * 216)) + (c_inner * 72)) + (((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused % 24) >> 3) * 24)) + (i_outer_outer_inner * 8)) + (b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused & 7))] = ((data[((((((b_outer_inner * 1080) + ((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused / 24) * 216)) + (c_inner * 72)) + (((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused % 24) >> 3) * 24)) + (i_outer_outer_inner * 8)) + (b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused & 7))] * Scale[(((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused / 24) * 3) + c_inner)]) + Shift[(((b_outer_outer_outer_c_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused / 24) * 3) + c_inner)]);
        }
      }
    }
  }
}


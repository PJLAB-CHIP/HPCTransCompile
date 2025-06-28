void default_function_kernel(float* Scale, float* ScaleShift, float* Shift, float* data) {
  #pragma omp parallel for
  for (int32_t b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused = 0; b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused < 340; ++b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused) {
    for (int32_t cb_outer_outer_inner = 0; cb_outer_outer_inner < 11; ++cb_outer_outer_inner) {
      for (int32_t cc_outer_inner = 0; cc_outer_inner < 2; ++cc_outer_inner) {
        for (int32_t i_outer_inner = 0; i_outer_inner < 7; ++i_outer_inner) {
          ScaleShift[((((((b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused / 20) * 3080) + (cc_outer_inner * 1540)) + (i_outer_inner * 220)) + ((b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused % 20) * 11)) + cb_outer_outer_inner)] = ((data[((((((b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused / 20) * 3080) + (cc_outer_inner * 1540)) + (i_outer_inner * 220)) + ((b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused % 20) * 11)) + cb_outer_outer_inner)] * Scale[((cc_outer_inner * 11) + cb_outer_outer_inner)]) + Shift[((cc_outer_inner * 11) + cb_outer_outer_inner)]);
        }
      }
    }
  }
}


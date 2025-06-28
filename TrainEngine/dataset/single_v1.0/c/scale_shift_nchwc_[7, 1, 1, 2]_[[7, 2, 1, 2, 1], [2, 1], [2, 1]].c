void default_function_kernel(float* Scale, float* ScaleShift, float* Shift, float* data) {
  for (int32_t j_outer_outer_outer = 0; j_outer_outer_outer < 2; ++j_outer_outer_outer) {
    for (int32_t b_outer_inner = 0; b_outer_inner < 7; ++b_outer_inner) {
      for (int32_t cc_inner = 0; cc_inner < 2; ++cc_inner) {
        ScaleShift[(((b_outer_inner * 4) + (cc_inner * 2)) + j_outer_outer_outer)] = ((data[(((b_outer_inner * 4) + (cc_inner * 2)) + j_outer_outer_outer)] * Scale[cc_inner]) + Shift[cc_inner]);
      }
    }
  }
}


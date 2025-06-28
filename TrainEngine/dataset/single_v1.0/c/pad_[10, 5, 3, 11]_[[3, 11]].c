void default_function_kernel(float* PadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 6; ++i0) {
    for (int32_t i1 = 0; i1 < 14; ++i1) {
      PadInput[((i0 * 14) + i1)] = (((((1 <= i0) && (i0 < 4)) && (2 <= i1)) && (i1 < 13)) ? data[(((i0 * 11) + i1) - 13)] : 0.000000e+00f);
    }
  }
}


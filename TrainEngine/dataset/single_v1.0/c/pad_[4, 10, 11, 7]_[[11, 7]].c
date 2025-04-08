void default_function_kernel(float* PadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 14; ++i0) {
    for (int32_t i1 = 0; i1 < 10; ++i1) {
      PadInput[((i0 * 10) + i1)] = (((((1 <= i0) && (i0 < 12)) && (2 <= i1)) && (i1 < 9)) ? data[(((i0 * 7) + i1) - 9)] : 0.000000e+00f);
    }
  }
}


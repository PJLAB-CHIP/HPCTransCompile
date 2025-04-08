void default_function_kernel(float* PadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 5; ++i0) {
    for (int32_t i1 = 0; i1 < 7; ++i1) {
      PadInput[((i0 * 7) + i1)] = (((((1 <= i0) && (i0 < 3)) && (2 <= i1)) && (i1 < 6)) ? data[(((i0 * 4) + i1) - 6)] : 0.000000e+00f);
    }
  }
}


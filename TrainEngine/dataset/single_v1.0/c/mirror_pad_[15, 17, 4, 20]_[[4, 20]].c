void default_function_kernel(float* MirrorPadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 7; ++i0) {
    for (int32_t i1 = 0; i1 < 23; ++i1) {
      MirrorPadInput[((i0 * 23) + i1)] = data[((((5 <= i0) ? (8 - i0) : ((i0 < 1) ? (0 - i0) : (i0 - 1))) * 20) + ((i1 == 22) ? (41 - i1) : ((i1 < 2) ? (1 - i1) : (i1 - 2))))];
    }
  }
}


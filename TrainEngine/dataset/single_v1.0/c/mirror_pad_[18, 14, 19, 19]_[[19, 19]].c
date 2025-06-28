void default_function_kernel(float* MirrorPadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 22; ++i0) {
    for (int32_t i1 = 0; i1 < 22; ++i1) {
      MirrorPadInput[((i0 * 22) + i1)] = data[((((20 <= i0) ? (38 - i0) : ((i0 < 1) ? (0 - i0) : (i0 - 1))) * 19) + ((i1 == 21) ? (39 - i1) : ((i1 < 2) ? (1 - i1) : (i1 - 2))))];
    }
  }
}


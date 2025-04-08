void default_function_kernel(float* MirrorPadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 77; ++i0_i1_fused) {
    MirrorPadInput[i0_i1_fused] = data[((((55 <= i0_i1_fused) ? (8 - (i0_i1_fused / 11)) : ((i0_i1_fused < 11) ? 0 : ((i0_i1_fused / 11) - 1))) * 8) + (((i0_i1_fused % 11) == 10) ? 7 : (((i0_i1_fused % 11) < 2) ? (1 - (i0_i1_fused % 11)) : ((i0_i1_fused % 11) - 2))))];
  }
}


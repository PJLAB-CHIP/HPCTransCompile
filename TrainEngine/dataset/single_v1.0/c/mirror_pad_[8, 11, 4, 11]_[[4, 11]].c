void default_function_kernel(float* MirrorPadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 98; ++i0_i1_fused) {
    MirrorPadInput[i0_i1_fused] = data[((((70 <= i0_i1_fused) ? (8 - (i0_i1_fused / 14)) : ((i0_i1_fused < 14) ? 0 : ((i0_i1_fused / 14) - 1))) * 11) + (((i0_i1_fused % 14) == 13) ? 10 : (((i0_i1_fused % 14) < 2) ? (1 - (i0_i1_fused % 14)) : ((i0_i1_fused % 14) - 2))))];
  }
}


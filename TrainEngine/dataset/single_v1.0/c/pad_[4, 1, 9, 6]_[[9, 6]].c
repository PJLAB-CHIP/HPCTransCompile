void default_function_kernel(float* PadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 108; ++i0_i1_fused) {
    PadInput[i0_i1_fused] = (((((9 <= i0_i1_fused) && (i0_i1_fused < 90)) && (2 <= (i0_i1_fused % 9))) && ((i0_i1_fused % 9) < 8)) ? data[((((i0_i1_fused / 9) * 6) + (i0_i1_fused % 9)) - 8)] : 0.000000e+00f);
  }
}


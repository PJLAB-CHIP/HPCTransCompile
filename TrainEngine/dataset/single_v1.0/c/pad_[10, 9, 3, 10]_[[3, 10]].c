void default_function_kernel(float* PadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 78; ++i0_i1_fused) {
    PadInput[i0_i1_fused] = (((((13 <= i0_i1_fused) && (i0_i1_fused < 52)) && (2 <= (i0_i1_fused % 13))) && ((i0_i1_fused % 13) < 12)) ? data[((((i0_i1_fused / 13) * 10) + (i0_i1_fused % 13)) - 12)] : 0.000000e+00f);
  }
}


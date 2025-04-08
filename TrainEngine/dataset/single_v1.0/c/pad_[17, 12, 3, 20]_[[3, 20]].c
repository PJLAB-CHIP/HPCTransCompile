void default_function_kernel(float* PadInput, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 138; ++i0_i1_fused) {
    PadInput[i0_i1_fused] = (((((23 <= i0_i1_fused) && (i0_i1_fused < 92)) && (2 <= (i0_i1_fused % 23))) && ((i0_i1_fused % 23) < 22)) ? data[((((i0_i1_fused / 23) * 20) + (i0_i1_fused % 23)) - 22)] : 0.000000e+00f);
  }
}


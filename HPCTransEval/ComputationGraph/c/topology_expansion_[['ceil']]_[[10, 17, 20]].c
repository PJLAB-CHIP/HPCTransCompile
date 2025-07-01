void default_function_kernel(float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 3400; ++i0_i1_fused_i2_fused) {
    compute[i0_i1_fused_i2_fused] = ceilf(ph_0[i0_i1_fused_i2_fused]);
  }
}


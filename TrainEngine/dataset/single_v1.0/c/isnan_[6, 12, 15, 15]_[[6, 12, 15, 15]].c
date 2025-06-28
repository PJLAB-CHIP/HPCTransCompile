void default_function_kernel(int8_t* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1080; ++i0_i1_fused_i2_fused) {
    for (int32_t i3_s = 0; i3_s < 15; ++i3_s) {
      compute[((i0_i1_fused_i2_fused * 15) + i3_s)] = ((int8_t)(data[((i0_i1_fused_i2_fused * 15) + i3_s)] != data[((i0_i1_fused_i2_fused * 15) + i3_s)]));
    }
  }
}


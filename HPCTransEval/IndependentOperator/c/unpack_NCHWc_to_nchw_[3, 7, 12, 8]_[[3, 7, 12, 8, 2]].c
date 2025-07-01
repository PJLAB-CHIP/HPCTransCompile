void default_function_kernel(float* output_unpack, float* packed_out) {
  #pragma omp parallel for
  for (int32_t n_c_fused_h_fused = 0; n_c_fused_h_fused < 504; ++n_c_fused_h_fused) {
    for (int32_t w = 0; w < 8; ++w) {
      output_unpack[((n_c_fused_h_fused * 8) + w)] = packed_out[(((((n_c_fused_h_fused / 24) * 192) + ((n_c_fused_h_fused % 12) * 16)) + (w * 2)) + ((n_c_fused_h_fused % 24) / 12))];
    }
  }
}


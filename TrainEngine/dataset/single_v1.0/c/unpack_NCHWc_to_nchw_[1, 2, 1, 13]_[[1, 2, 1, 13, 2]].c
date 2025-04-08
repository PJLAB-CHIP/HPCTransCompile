void default_function_kernel(float* output_unpack, float* packed_out) {
  for (int32_t c = 0; c < 4; ++c) {
    for (int32_t w = 0; w < 13; ++w) {
      output_unpack[((c * 13) + w)] = packed_out[((((c >> 1) * 26) + (w * 2)) + (c & 1))];
    }
  }
}


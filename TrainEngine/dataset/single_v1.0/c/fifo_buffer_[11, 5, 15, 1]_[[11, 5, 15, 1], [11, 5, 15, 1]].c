void default_function_kernel(float* data, float* new_buffer) {
  #pragma omp parallel for
  for (int32_t i_j_fused = 0; i_j_fused < 55; ++i_j_fused) {
    for (int32_t k = 0; k < 15; ++k) {
      new_buffer[((i_j_fused * 15) + k)] = data[((i_j_fused * 15) + k)];
    }
  }
}


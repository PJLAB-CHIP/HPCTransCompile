void default_function_kernel(float* data, float* new_buffer) {
  #pragma omp parallel for
  for (int32_t i = 0; i < 3; ++i) {
    for (int32_t j = 0; j < 15; ++j) {
      for (int32_t k = 0; k < 6; ++k) {
        for (int32_t l = 0; l < 18; ++l) {
          new_buffer[((((i * 1620) + (j * 108)) + (k * 18)) + l)] = data[((((i * 1620) + (j * 108)) + (k * 18)) + l)];
        }
      }
    }
  }
}


void default_function_kernel(float* data, float* new_buffer) {
  #pragma omp parallel for
  for (int32_t i = 0; i < 10; ++i) {
    for (int32_t j = 0; j < 12; ++j) {
      for (int32_t k = 0; k < 16; ++k) {
        for (int32_t l = 0; l < 19; ++l) {
          new_buffer[((((i * 3648) + (j * 304)) + (k * 19)) + l)] = data[((((i * 3648) + (j * 304)) + (k * 19)) + l)];
        }
      }
    }
  }
}


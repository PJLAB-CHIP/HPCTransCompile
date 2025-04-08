void default_function_kernel(float* data, float* new_buffer) {
  #pragma omp parallel for
  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < 17; ++j) {
      for (int32_t k = 0; k < 16; ++k) {
        for (int32_t l = 0; l < 16; ++l) {
          new_buffer[((((i * 4352) + (j * 256)) + (k * 16)) + l)] = data[((((i * 4352) + (j * 256)) + (k * 16)) + l)];
        }
      }
    }
  }
}


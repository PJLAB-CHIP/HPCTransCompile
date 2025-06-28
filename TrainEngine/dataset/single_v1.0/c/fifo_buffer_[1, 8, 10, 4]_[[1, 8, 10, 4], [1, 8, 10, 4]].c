void default_function_kernel(float* data, float* new_buffer) {
  for (int32_t j = 0; j < 8; ++j) {
    for (int32_t k = 0; k < 10; ++k) {
      for (int32_t l = 0; l < 4; ++l) {
        new_buffer[(((j * 40) + (k * 4)) + l)] = data[(((j * 40) + (k * 4)) + l)];
      }
    }
  }
}


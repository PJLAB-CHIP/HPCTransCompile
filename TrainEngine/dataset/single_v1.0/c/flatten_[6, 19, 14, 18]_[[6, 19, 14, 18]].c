void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i = 0; i < 6; ++i) {
    for (int32_t j = 0; j < 4788; ++j) {
      compute[((i * 4788) + j)] = data[((i * 4788) + j)];
    }
  }
}


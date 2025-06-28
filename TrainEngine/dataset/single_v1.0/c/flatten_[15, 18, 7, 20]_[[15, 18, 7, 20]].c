void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i = 0; i < 15; ++i) {
    for (int32_t j = 0; j < 2520; ++j) {
      compute[((i * 2520) + j)] = data[((i * 2520) + j)];
    }
  }
}


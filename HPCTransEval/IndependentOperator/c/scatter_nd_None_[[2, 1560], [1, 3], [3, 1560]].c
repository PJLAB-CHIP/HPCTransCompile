void default_function_kernel(float* data, int32_t* indices, float* scatter_nd_generic, float* updates) {
  for (int32_t i = 0; i < 3120; ++i) {
    scatter_nd_generic[i] = data[i];
  }
  for (int32_t j = 0; j < 3; ++j) {
    #pragma omp parallel for
    for (int32_t k = 0; k < 1560; ++k) {
      scatter_nd_generic[((indices[j] * 1560) + k)] = (scatter_nd_generic[((indices[j] * 1560) + k)] + updates[((j * 1560) + k)]);
    }
  }
}


void default_function_kernel(float* Im, float* Re, float* dft_cpu, float* dft_cpu_1) {
  #pragma omp parallel for
  for (int32_t i = 0; i < 1200; ++i) {
    for (int32_t j = 0; j < 50; ++j) {
      dft_cpu[((i * 50) + j)] = 0.000000e+00f;
      dft_cpu_1[((i * 50) + j)] = 0.000000e+00f;
      for (int32_t k = 0; k < 50; ++k) {
        float cse_var_2 = (((-6.283185e+00f * ((float)j)) * 2.000000e-02f) * ((float)k));
        dft_cpu[((i * 50) + j)] = (dft_cpu[((i * 50) + j)] + ((Re[((i * 50) + k)] * cosf(cse_var_2)) - (Im[((i * 50) + k)] * sinf(cse_var_2))));
        dft_cpu_1[((i * 50) + j)] = (dft_cpu_1[((i * 50) + j)] + ((Re[((i * 50) + k)] * sinf(cse_var_2)) + (Im[((i * 50) + k)] * cosf(cse_var_2))));
      }
    }
  }
}


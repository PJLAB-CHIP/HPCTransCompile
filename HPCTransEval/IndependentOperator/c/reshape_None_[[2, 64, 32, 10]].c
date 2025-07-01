void default_function_kernel(float* A, float* T_reshape) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 128; ++ax0) {
    for (int32_t ax1_outer = 0; ax1_outer < 20; ++ax1_outer) {
      for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
        T_reshape[(((ax0 * 320) + (ax1_outer * 16)) + ax1_inner)] = A[(((ax0 * 320) + (ax1_outer * 16)) + ax1_inner)];
      }
    }
  }
}


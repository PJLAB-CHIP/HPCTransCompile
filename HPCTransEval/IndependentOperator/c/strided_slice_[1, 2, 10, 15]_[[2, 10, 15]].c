void default_function_kernel(float* T_strided_slice, float* a) {
  for (int32_t ax1 = 0; ax1 < 5; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 7; ++ax2) {
      T_strided_slice[((ax1 * 7) + ax2)] = a[(((ax1 * 15) + ax2) + 183)];
    }
  }
}


void default_function_kernel(float* T_add, float* data, float* data_1) {
  for (int32_t ax2 = 0; ax2 < 4; ++ax2) {
    for (int32_t ax3 = 0; ax3 < 3; ++ax3) {
      T_add[((ax2 * 3) + ax3)] = (data[((ax2 * 3) + ax3)] + data_1[((ax2 * 3) + ax3)]);
    }
  }
}


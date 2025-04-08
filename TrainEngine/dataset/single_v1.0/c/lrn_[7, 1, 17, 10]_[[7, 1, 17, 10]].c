void default_function_kernel(float* T_divide, float* data) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 7; ++ax0) {
    float tensor[170];
    for (int32_t ax2 = 0; ax2 < 17; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 10; ++ax3) {
        tensor[((ax2 * 10) + ax3)] = 0.000000e+00f;
        tensor[((ax2 * 10) + ax3)] = (tensor[((ax2 * 10) + ax3)] + (data[(((ax0 * 170) + (ax2 * 10)) + ax3)] * data[(((ax0 * 170) + (ax2 * 10)) + ax3)]));
      }
    }
    for (int32_t ax2_1 = 0; ax2_1 < 17; ++ax2_1) {
      for (int32_t ax3_1 = 0; ax3_1 < 10; ++ax3_1) {
        T_divide[(((ax0 * 170) + (ax2_1 * 10)) + ax3_1)] = (data[(((ax0 * 170) + (ax2_1 * 10)) + ax3_1)] / powf((2.000000e+00f + (1.000000e-04f * tensor[((ax2_1 * 10) + ax3_1)])), 7.500000e-01f));
      }
    }
  }
}


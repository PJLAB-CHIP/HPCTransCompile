void default_function_kernel(float* data, int32_t* extern_1) {
  float temp_data[1];
  int32_t temp_index[1];
  #pragma omp parallel for
  for (int32_t i = 0; i < 64; ++i) {
    for (int32_t j = 0; j < 256; ++j) {
      extern_1[((i * 512) + (j * 2))] = (j * 2);
    }
    for (int32_t k = 0; k < 256; ++k) {
      extern_1[(((i * 512) + (k * 2)) + 1)] = ((k * 2) + 1);
    }
    for (int32_t i_0 = 0; i_0 < 512; ++i_0) {
      for (int32_t i_1 = 0; i_1 < 256; ++i_1) {
        if (((((i * 512) + (i_1 * 2)) + (i_0 & 1)) < 511) && (data[(((i * 512) + (i_1 * 2)) + (i_0 & 1))] < data[((((i * 512) + (i_1 * 2)) + (i_0 & 1)) + 1)])) {
          temp_data[0] = data[(((i * 512) + (i_1 * 2)) + (i_0 & 1))];
          data[(((i * 512) + (i_1 * 2)) + (i_0 & 1))] = data[((((i * 512) + (i_1 * 2)) + (i_0 & 1)) + 1)];
          data[((((i * 512) + (i_1 * 2)) + (i_0 & 1)) + 1)] = temp_data[0];
          temp_index[0] = extern_1[(((i * 512) + (i_1 * 2)) + (i_0 & 1))];
          extern_1[(((i * 512) + (i_1 * 2)) + (i_0 & 1))] = extern_1[((((i * 512) + (i_1 * 2)) + (i_0 & 1)) + 1)];
          extern_1[((((i * 512) + (i_1 * 2)) + (i_0 & 1)) + 1)] = temp_index[0];
        }
      }
    }
  }
}


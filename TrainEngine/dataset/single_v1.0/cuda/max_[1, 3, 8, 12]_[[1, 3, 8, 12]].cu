extern "C" __global__ void default_function_kernel(float* __restrict__ data, float* __restrict__ data_red) {
  data_red[0] = -3.402823e+38f;
  for (int k1 = 0; k1 < 3; ++k1) {
    for (int k2 = 0; k2 < 8; ++k2) {
      for (int k3 = 0; k3 < 12; ++k3) {
        data_red[0] = max(data_red[0], data[(((k1 * 96) + (k2 * 12)) + k3)]);
      }
    }
  }
}


extern "C" __global__ void default_function_kernel(float* __restrict__ data, float* __restrict__ data_red) {
  data_red[0] = 0.000000e+00f;
  for (int k0 = 0; k0 < 20; ++k0) {
    for (int k1 = 0; k1 < 14; ++k1) {
      for (int k2 = 0; k2 < 2; ++k2) {
        for (int k3 = 0; k3 < 9; ++k3) {
          data_red[0] = (data_red[0] + data[((((k0 * 252) + (k1 * 18)) + (k2 * 9)) + k3)]);
        }
      }
    }
  }
}


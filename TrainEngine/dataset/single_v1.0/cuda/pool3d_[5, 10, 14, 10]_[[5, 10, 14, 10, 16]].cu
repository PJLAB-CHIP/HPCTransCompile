extern "C" __global__ void __launch_bounds__(20) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      for (int rv2 = 0; rv2 < 3; ++rv2) {
        pool_max[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))], ((((1 <= ((((((int)blockIdx.x) % 14) >> 1) * 2) + rv0)) && (1 <= ((((((((int)blockIdx.x) & 1) * 5) + (((int)threadIdx.x) >> 2)) >> 1) * 2) + rv1))) && (1 <= (((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 7) * 2) + rv2))) ? data[((((((((((int)blockIdx.x) >> 1) * 320) + (rv0 * 160)) + (((((((int)blockIdx.x) & 1) * 5) + (((int)threadIdx.x) >> 2)) >> 1) * 32)) + (rv1 * 16)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 7) * 2)) + rv2) - 177)] : -3.402823e+38f));
      }
    }
  }
}


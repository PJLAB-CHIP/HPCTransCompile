extern "C" __global__ void __launch_bounds__(40) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      for (int rv2 = 0; rv2 < 3; ++rv2) {
        pool_max[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))], ((((((1 <= ((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) % 24) / 6) * 2) + rv0)) && ((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) % 24) / 6) + (rv0 >> 1)) < 4)) && (1 <= (((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) % 6) * 2) + rv1))) && (((rv1 >> 1) + (((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) % 6)) < 6)) && (1 <= (((((int)threadIdx.x) & 7) * 2) + rv2))) ? data[(((((((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) / 24) * 1232) + (((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) % 24) / 6) * 352)) + (rv0 * 176)) + ((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) % 6) * 32)) + (rv1 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + rv2) - 193)] : -3.402823e+38f));
      }
    }
  }
}


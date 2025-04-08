extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      for (int rv2 = 0; rv2 < 3; ++rv2) {
        pool_max[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))], (((((1 <= ((((((int)blockIdx.x) & 1) * 4) + ((((int)threadIdx.x) >> 5) * 2)) + rv0)) && (((((((int)threadIdx.x) >> 5) + (rv0 >> 1)) >> 1) + (((int)blockIdx.x) & 1)) < 2)) && (1 <= ((((((int)threadIdx.x) & 31) >> 3) * 2) + rv1))) && (1 <= (((((int)threadIdx.x) & 7) * 2) + rv2))) ? data[((((((((((((int)blockIdx.x) >> 1) * 896) + ((((int)blockIdx.x) & 1) * 512)) + ((((int)threadIdx.x) >> 5) * 256)) + (rv0 * 128)) + (((((int)threadIdx.x) & 31) >> 3) * 32)) + (rv1 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + rv2) - 145)] : -3.402823e+38f));
      }
    }
  }
}


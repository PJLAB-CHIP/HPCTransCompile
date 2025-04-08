extern "C" __global__ void __launch_bounds__(24) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      for (int rv2 = 0; rv2 < 3; ++rv2) {
        pool_max[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))], ((((((1 <= rv0) && (rv0 < 2)) && (1 <= (((((((int)threadIdx.x) >> 3) + ((int)blockIdx.x)) & 1) * 2) + rv1))) && (((rv1 >> 1) + (((((int)threadIdx.x) >> 3) + ((int)blockIdx.x)) & 1)) < 2)) && (1 <= (((((int)threadIdx.x) & 7) * 2) + rv2))) ? data[((((((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 3)) >> 1) * 48) + (rv0 * 48)) + ((((((int)threadIdx.x) >> 3) + ((int)blockIdx.x)) & 1) * 32)) + (rv1 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + rv2) - 65)] : -3.402823e+38f));
      }
    }
  }
}


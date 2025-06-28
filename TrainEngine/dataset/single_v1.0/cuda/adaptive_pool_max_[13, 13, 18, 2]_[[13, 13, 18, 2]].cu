extern "C" __global__ void __launch_bounds__(26) default_function_kernel(float* __restrict__ adaptive_pool_max, float* __restrict__ data) {
  adaptive_pool_max[((((int)blockIdx.x) * 26) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < ((((((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 2) + 2) % 8) == 0) ? (((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 9) + 9) >> 2) : ((((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 9) + 9) >> 2) + 1)) - ((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 18) >> 3)); ++rv0) {
    for (int rv1 = 0; rv1 < (((((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) & 7) * 2) + 2) % 8) == 0) ? (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) & 7) + 1) >> 2) : ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) & 7) + 1) >> 2) + 1)) - ((((((int)threadIdx.x) >> 1) + ((int)blockIdx.x)) & 3) >> 1)); ++rv1) {
      adaptive_pool_max[((((int)blockIdx.x) * 26) + ((int)threadIdx.x))] = max(adaptive_pool_max[((((int)blockIdx.x) * 26) + ((int)threadIdx.x))], data[((((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 1)) >> 5) * 36) + (((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 18) >> 3) * 2)) + (rv0 * 2)) + ((((((int)threadIdx.x) >> 1) + ((int)blockIdx.x)) & 3) >> 1)) + rv1)]);
    }
  }
}


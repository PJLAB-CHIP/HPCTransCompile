extern "C" __global__ void __launch_bounds__(34) default_function_kernel(float* __restrict__ adaptive_pool_max, float* __restrict__ data) {
  adaptive_pool_max[((((int)blockIdx.x) * 34) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < ((((((((((((int)blockIdx.x) * 17) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 4) + 4) % 8) == 0) ? (((((((((int)blockIdx.x) * 17) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 3) + 3) >> 1) : ((((((((((int)blockIdx.x) * 17) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 3) + 3) >> 1) + 1)) - ((((((((int)blockIdx.x) * 17) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 12) >> 3)); ++rv0) {
    for (int rv1 = 0; rv1 < (((((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) & 7) * 3) + 3) % 8) == 0) ? ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) & 7) * 3) + 3) >> 3) : (((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) & 7) * 3) + 3) >> 3) + 1)) - (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) & 7) * 3) >> 3)); ++rv1) {
      adaptive_pool_max[((((int)blockIdx.x) * 34) + ((int)threadIdx.x))] = max(adaptive_pool_max[((((int)blockIdx.x) * 34) + ((int)threadIdx.x))], data[((((((((((int)blockIdx.x) * 17) + (((int)threadIdx.x) >> 1)) >> 5) * 36) + (((((((((int)blockIdx.x) * 17) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 12) >> 3) * 3)) + (rv0 * 3)) + (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) & 7) * 3) >> 3)) + rv1)]);
    }
  }
}


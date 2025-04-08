extern "C" __global__ void __launch_bounds__(40) default_function_kernel(float* __restrict__ adaptive_pool_max, float* __restrict__ data) {
  adaptive_pool_max[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < (((((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) & 7) * 6) + 6) % 8) == 0) ? ((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) & 7) * 3) + 3) >> 2) : (((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) & 7) * 3) + 3) >> 2) + 1)) - (((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) & 7) * 6) >> 3)); ++rv0) {
    for (int rv1 = 0; rv1 < (((((((((int)threadIdx.x) & 7) * 4) + 4) % 8) == 0) ? ((((((int)threadIdx.x) & 7) * 5) + 5) >> 1) : (((((((int)threadIdx.x) & 7) * 5) + 5) >> 1) + 1)) - (((((int)threadIdx.x) & 7) * 20) >> 3)); ++rv1) {
      adaptive_pool_max[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = max(adaptive_pool_max[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))], data[((((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) >> 3) * 120) + ((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) & 7) * 6) >> 3) * 20)) + (rv0 * 20)) + (((((int)threadIdx.x) & 7) * 20) >> 3)) + rv1)]);
    }
  }
}


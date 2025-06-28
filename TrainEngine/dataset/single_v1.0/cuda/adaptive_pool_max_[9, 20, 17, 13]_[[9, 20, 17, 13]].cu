extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ adaptive_pool_max, float* __restrict__ data) {
  adaptive_pool_max[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < ((((((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) & 7) + 1) % 8) == 0) ? ((((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) & 7) * 17) + 17) >> 3) : (((((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) & 7) * 17) + 17) >> 3) + 1)) - ((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) & 7) * 2)); ++rv0) {
    for (int rv1 = 0; rv1 < (((((((((int)threadIdx.x) & 7) * 5) + 5) % 8) == 0) ? ((((((int)threadIdx.x) & 7) * 13) + 13) >> 3) : (((((((int)threadIdx.x) & 7) * 13) + 13) >> 3) + 1)) - (((((int)threadIdx.x) & 7) * 13) >> 3)); ++rv1) {
      adaptive_pool_max[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = max(adaptive_pool_max[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))], data[((((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 4)) >> 2) * 221) + ((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) & 7) * 26)) + (rv0 * 13)) + (((((int)threadIdx.x) & 7) * 13) >> 3)) + rv1)]);
    }
  }
}


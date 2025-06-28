extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ adaptive_pool_max, float* __restrict__ data) {
  adaptive_pool_max[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < (((((((((int)blockIdx.x) & 7) * 4) + 4) % 8) == 0) ? ((((((int)blockIdx.x) & 7) * 5) + 5) >> 1) : (((((((int)blockIdx.x) & 7) * 5) + 5) >> 1) + 1)) - (((((int)blockIdx.x) & 7) * 20) >> 3)); ++rv0) {
    for (int rv1 = 0; rv1 < ((((((((int)threadIdx.x) * 2) + 2) % 8) == 0) ? (((((int)threadIdx.x) * 9) + 9) >> 2) : ((((((int)threadIdx.x) * 9) + 9) >> 2) + 1)) - ((((int)threadIdx.x) * 18) >> 3)); ++rv1) {
      adaptive_pool_max[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = max(adaptive_pool_max[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))], data[((((((((int)blockIdx.x) >> 3) * 360) + ((((((int)blockIdx.x) & 7) * 20) >> 3) * 18)) + (rv0 * 18)) + ((((int)threadIdx.x) * 18) >> 3)) + rv1)]);
    }
  }
}


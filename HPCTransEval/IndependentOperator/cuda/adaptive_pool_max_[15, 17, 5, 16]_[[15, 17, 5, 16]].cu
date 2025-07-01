extern "C" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ adaptive_pool_max, float* __restrict__ data) {
  adaptive_pool_max[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < ((((((((((((int)blockIdx.x) * 15) + (((int)threadIdx.x) >> 2)) & 15) >> 1) * 5) + 5) % 8) == 0) ? (((((((((int)blockIdx.x) * 15) + (((int)threadIdx.x) >> 2)) & 15) >> 1) * 5) + 5) >> 3) : ((((((((((int)blockIdx.x) * 15) + (((int)threadIdx.x) >> 2)) & 15) >> 1) * 5) + 5) >> 3) + 1)) - ((((((((int)blockIdx.x) * 15) + (((int)threadIdx.x) >> 2)) & 15) >> 1) * 5) >> 3)); ++rv0) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      adaptive_pool_max[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = max(adaptive_pool_max[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))], data[((((((((((int)blockIdx.x) * 15) + (((int)threadIdx.x) >> 2)) >> 4) * 80) + (((((((((int)blockIdx.x) * 15) + (((int)threadIdx.x) >> 2)) & 15) >> 1) * 5) >> 3) * 16)) + (rv0 * 16)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 7) * 2)) + rv1)]);
    }
  }
}


extern "C" __global__ void __launch_bounds__(38) default_function_kernel(float* __restrict__ adaptive_pool_max, float* __restrict__ data) {
  adaptive_pool_max[((((int)blockIdx.x) * 38) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < ((((((((((((int)blockIdx.x) * 19) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 4) + 4) % 8) == 0) ? (((((((((int)blockIdx.x) * 19) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 5) + 5) >> 1) : ((((((((((int)blockIdx.x) * 19) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 5) + 5) >> 1) + 1)) - ((((((((int)blockIdx.x) * 19) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 20) >> 3)); ++rv0) {
    for (int rv1 = 0; rv1 < (((((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) & 7) * 3) + 3) % 8) == 0) ? ((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) & 7) * 19) + 19) >> 3) : (((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) & 7) * 19) + 19) >> 3) + 1)) - (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) & 7) * 19) >> 3)); ++rv1) {
      adaptive_pool_max[((((int)blockIdx.x) * 38) + ((int)threadIdx.x))] = max(adaptive_pool_max[((((int)blockIdx.x) * 38) + ((int)threadIdx.x))], data[((((((((((int)blockIdx.x) * 19) + (((int)threadIdx.x) >> 1)) >> 5) * 380) + (((((((((int)blockIdx.x) * 19) + (((int)threadIdx.x) >> 1)) & 31) >> 2) * 20) >> 3) * 19)) + (rv0 * 19)) + (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) & 7) * 19) >> 3)) + rv1)]);
    }
  }
}


extern "C" __global__ void __launch_bounds__(4) default_function_kernel(float* __restrict__ adaptive_pool_max, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) < 49) {
    adaptive_pool_max[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = -3.402823e+38f;
  }
  for (int rv0 = 0; rv0 < 18; ++rv0) {
    for (int rv1 = 0; rv1 < 6; ++rv1) {
      if (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) < 49) {
        adaptive_pool_max[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = max(adaptive_pool_max[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))], data[((((((int)blockIdx.x) * 432) + (((int)threadIdx.x) * 108)) + (rv0 * 6)) + rv1)]);
      }
    }
  }
}


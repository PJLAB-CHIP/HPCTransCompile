extern "C" __global__ void __launch_bounds__(14) default_function_kernel(float* __restrict__ adaptive_pool_max, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 14) + ((int)threadIdx.x)) < 323) {
    adaptive_pool_max[((((int)blockIdx.x) * 14) + ((int)threadIdx.x))] = -3.402823e+38f;
  }
  for (int rv0 = 0; rv0 < 12; ++rv0) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      if (((((int)blockIdx.x) * 14) + ((int)threadIdx.x)) < 323) {
        adaptive_pool_max[((((int)blockIdx.x) * 14) + ((int)threadIdx.x))] = max(adaptive_pool_max[((((int)blockIdx.x) * 14) + ((int)threadIdx.x))], data[((((((int)blockIdx.x) * 336) + (((int)threadIdx.x) * 24)) + (rv0 * 2)) + rv1)]);
      }
    }
  }
}


extern "C" __global__ void __launch_bounds__(6) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 6) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_max[((((int)blockIdx.x) * 6) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 6) + ((int)threadIdx.x))], (((1 <= (((((int)blockIdx.x) % 9) * 2) + rv0)) && (1 <= ((((int)threadIdx.x) * 2) + rv1))) ? data[(((((((int)blockIdx.x) * 24) + (rv0 * 12)) + (((int)threadIdx.x) * 2)) + rv1) - 13)] : -3.402823e+38f));
    }
  }
}


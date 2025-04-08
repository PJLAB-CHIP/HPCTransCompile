extern "C" __global__ void __launch_bounds__(39) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 39) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_max[((((int)blockIdx.x) * 39) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 39) + ((int)threadIdx.x))], ((((1 <= ((((((((int)threadIdx.x) / 3) + ((int)blockIdx.x)) % 6) / 3) * 2) + rv0)) && ((((((((int)threadIdx.x) / 3) + ((int)blockIdx.x)) % 6) / 3) + (rv0 >> 1)) < 2)) && (1 <= (((((((int)blockIdx.x) * 3) + ((int)threadIdx.x)) % 9) * 2) + rv1))) ? data[(((((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) / 3)) / 6) * 54) + (((((((int)threadIdx.x) / 3) + ((int)blockIdx.x)) % 6) / 3) * 36)) + (rv0 * 18)) + ((((((int)blockIdx.x) * 3) + ((int)threadIdx.x)) % 9) * 2)) + rv1) - 19)] : -3.402823e+38f));
    }
  }
}


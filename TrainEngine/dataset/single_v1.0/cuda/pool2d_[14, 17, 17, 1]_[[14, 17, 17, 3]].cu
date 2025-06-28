extern "C" __global__ void __launch_bounds__(34) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 34) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_max[((((int)blockIdx.x) * 34) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 34) + ((int)threadIdx.x))], (((((1 <= (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) % 9) * 2) + rv0)) && (((rv0 >> 1) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) % 9)) < 9)) && (1 <= (((((int)threadIdx.x) & 1) * 2) + rv1))) && (((rv1 >> 1) + (((int)threadIdx.x) & 1)) < 2)) ? data[(((((((((((int)blockIdx.x) * 17) + (((int)threadIdx.x) >> 1)) / 9) * 51) + ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) % 9) * 6)) + (rv0 * 3)) + ((((int)threadIdx.x) & 1) * 2)) + rv1) - 4)] : -3.402823e+38f));
    }
  }
}


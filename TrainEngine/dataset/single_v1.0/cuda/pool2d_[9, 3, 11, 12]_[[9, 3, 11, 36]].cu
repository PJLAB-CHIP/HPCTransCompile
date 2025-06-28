extern "C" __global__ void __launch_bounds__(54) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_max[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))], ((((1 <= ((((((int)blockIdx.x) & 1) * 6) + ((((int)threadIdx.x) / 18) * 2)) + rv0)) && (((((((int)threadIdx.x) / 18) + (rv0 >> 1)) / 3) + (((int)blockIdx.x) & 1)) < 2)) && (1 <= (((((int)threadIdx.x) % 18) * 2) + rv1))) ? data[((((((((((int)blockIdx.x) >> 1) * 396) + ((((int)blockIdx.x) & 1) * 216)) + ((((int)threadIdx.x) / 18) * 72)) + (rv0 * 36)) + ((((int)threadIdx.x) % 18) * 2)) + rv1) - 37)] : -3.402823e+38f));
    }
  }
}


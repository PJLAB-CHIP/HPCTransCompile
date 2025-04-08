extern "C" __global__ void __launch_bounds__(40) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_max[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))], ((((1 <= ((((((((int)blockIdx.x) % 3) * 8) + (((int)threadIdx.x) / 5)) / 3) * 2) + rv0)) && ((((((((int)blockIdx.x) % 3) * 8) + (((int)threadIdx.x) / 5)) / 3) + (rv0 >> 1)) < 8)) && (1 <= (((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 15) * 2) + rv1))) ? data[(((((((((int)blockIdx.x) / 3) * 450) + (((((((int)blockIdx.x) % 3) * 8) + (((int)threadIdx.x) / 5)) / 3) * 60)) + (rv0 * 30)) + ((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 15) * 2)) + rv1) - 31)] : -3.402823e+38f));
    }
  }
}


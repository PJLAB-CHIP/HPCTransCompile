extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    pool_max[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], (((1 <= (((((int)threadIdx.x) & 3) * 2) + rv0)) && (((rv0 >> 1) + (((int)threadIdx.x) & 3)) < 4)) ? data[(((((((int)blockIdx.x) * 56) + ((((int)threadIdx.x) >> 2) * 7)) + ((((int)threadIdx.x) & 3) * 2)) + rv0) - 1)] : -3.402823e+38f));
  }
}


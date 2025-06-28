extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 2)) < 135) {
    pool_max[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = -3.402823e+38f;
  }
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 2)) < 135) {
      pool_max[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))], (((1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 3) * 2) + rv0)) && (((rv0 >> 1) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 3)) < 3)) ? data[(((((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) / 3) * 5) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 3) * 2)) + rv0) - 1)] : -3.402823e+38f));
    }
  }
}


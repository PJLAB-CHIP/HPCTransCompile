extern "C" __global__ void __launch_bounds__(50) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_max[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))], (((((1 <= ((((((((int)blockIdx.x) * 25) + (((int)threadIdx.x) >> 1)) % 130) / 13) * 2) + rv0)) && ((((((((int)blockIdx.x) * 25) + (((int)threadIdx.x) >> 1)) % 130) / 13) + (rv0 >> 1)) < 10)) && (1 <= (((((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 26) * 2) + rv1))) && (((rv1 >> 1) + (((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 26)) < 26)) ? data[(((((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 10)) / 26) * 969) + (((((((int)blockIdx.x) * 25) + (((int)threadIdx.x) >> 1)) % 130) / 13) * 102)) + (rv0 * 51)) + ((((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 26) * 2)) + rv1) - 52)] : -3.402823e+38f));
    }
  }
}


extern "C" __global__ void __launch_bounds__(24) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_max[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))], ((((1 <= ((((((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 51) / 17) * 2) + rv0)) && (1 <= (((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) % 17) * 2) + rv1))) && (((rv1 >> 1) + (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) % 17)) < 17)) ? data[((((((((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) / 17) * 66) + (rv0 * 33)) + ((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) % 17) * 2)) + rv1) - 34)] : -3.402823e+38f));
    }
  }
}


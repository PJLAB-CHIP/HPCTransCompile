extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      for (int rv2 = 0; rv2 < 3; ++rv2) {
        pool_max[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))], (((((1 <= ((((((((int)blockIdx.x) % 7) * 2) + (((int)threadIdx.x) >> 3)) / 7) * 2) + rv0)) && ((((((((int)blockIdx.x) % 7) * 2) + (((int)threadIdx.x) >> 3)) / 7) + (rv0 >> 1)) < 2)) && (1 <= (((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 3)) % 7) * 2) + rv1))) && (1 <= (((((int)threadIdx.x) & 7) * 2) + rv2))) ? data[(((((((((((int)blockIdx.x) / 7) * 672) + (((((((int)blockIdx.x) % 7) * 2) + (((int)threadIdx.x) >> 3)) / 7) * 448)) + (rv0 * 224)) + ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 3)) % 7) * 32)) + (rv1 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + rv2) - 241)] : -3.402823e+38f));
      }
    }
  }
}


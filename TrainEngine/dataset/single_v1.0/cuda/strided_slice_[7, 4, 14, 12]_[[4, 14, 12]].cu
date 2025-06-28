extern "C" __global__ void __launch_bounds__(24) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ a) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) / 3)) < 35) {
    T_strided_slice[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = a[(((((((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) / 35) * 168) + (((((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 35) / 7) * 12)) + (((((int)blockIdx.x) * 3) + ((int)threadIdx.x)) % 7)) + 195)];
  }
}


extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ a) {
  if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 105) {
    T_strided_slice[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = a[(((((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) / 35) * 140) + (((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 35) / 7) * 10)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 163)];
  }
}

